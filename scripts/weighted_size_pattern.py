import argparse
import csv
import hashlib
import json
import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mercury.checkpoint import checkpoint_tables
from mercury.constants import MODULATION_ID_ORDER
from mercury.data import collate_variable, generate_feasible_instance
from mercury.metrics import Jmean, inequality_violation, kkt_metrics, projected_stationarity_residual
from mercury.model import MercuryFoundationSolver
from mercury.utils import logspace_gamma_grid, seed_all


CHECKPOINT = (
    ROOT
    / "checkpoints"
    / "awf_projected_n_seed123_final.pt"
)
DEFAULT_OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "weighted_objective_size_pattern_n10"
)
SIZES = (32, 64, 128, 256, 512, 1024)
PATTERNS = ("random_positive", "grouped", "wide_two_level")
DEFAULT_INSTANCES_PER_SIZE = 10
INSTANCE_SEED = 20260921
WEIGHT_SEED = 20261021
T = 64
WARM_FEAS_STEPS = 16
ETA_EVAL = 0.01


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_tensor_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def make_weights(pattern: str, m: int, instance_id: int, device: torch.device):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(WEIGHT_SEED + 10000 * m + instance_id)
    if pattern == "random_positive":
        weights = 0.5 + torch.rand(m, generator=generator)
        weights = weights / weights.mean()
    elif pattern == "grouped":
        weights = torch.cat(
            [torch.full((m // 2,), 0.5), torch.full((m - m // 2,), 1.5)]
        )
        if instance_id % 2:
            weights = torch.flip(weights, dims=(0,))
    elif pattern == "wide_two_level":
        weights = torch.cat(
            [torch.full((m // 2,), 0.25), torch.full((m - m // 2,), 1.75)]
        )
        weights = weights[torch.randperm(m, generator=generator)]
    else:
        raise ValueError(f"Unknown pattern: {pattern}")
    weights = weights.view(1, m).to(device=device, dtype=torch.float32)
    if not torch.allclose(weights.mean(), torch.tensor(1.0, device=device), atol=1e-7):
        raise RuntimeError(f"Weight mean is not one for {pattern}, m={m}, instance={instance_id}")
    return weights


@torch.no_grad()
def metrics_for_state(model, pb, p, n, mu, weights):
    kkt = kkt_metrics(model, pb, p, n, None, mu, utility_weights=weights)
    return {
        "weighted_J": float(Jmean(model, pb, p, n, utility_weights=weights).item()),
        "inequality": float(inequality_violation(pb, p).item()),
        "weighted_KKT_p": float(kkt["kkt_p_active"]),
        "weighted_KKT_n": float(kkt["kkt_n"]),
        "weighted_R_stat": float(
            projected_stationarity_residual(
                model,
                pb,
                p,
                n,
                mu,
                stat_step=ETA_EVAL,
                utility_weights=weights,
            ).item()
        ),
        "sum_p_error": float((p.sum(-1, keepdim=True) - pb.P).abs().max().item()),
        "sum_n_error": float((n.sum(-1, keepdim=True) - pb.N).abs().max().item()),
        "all_finite": bool(all(torch.isfinite(x).all().item() for x in (p, n, mu))),
    }


def summarize_group(rows):
    adapted = [row for row in rows if row["mode"] == "weighted_gradient_adaptation"]
    paired = []
    for adapted_row in adapted:
        control_row = next(
            row
            for row in rows
            if row["mode"] == "uniform_gradient_control"
            and row["m"] == adapted_row["m"]
            and row["pattern"] == adapted_row["pattern"]
            and row["instance_id"] == adapted_row["instance_id"]
        )
        paired.append((control_row, adapted_row))
    rstat = [adapted_row["weighted_R_stat"] for _, adapted_row in paired]
    ratio = [
        adapted_row["weighted_R_stat"] / max(control_row["weighted_R_stat"], 1e-12)
        for control_row, adapted_row in paired
    ]
    return {
        "n": len(paired),
        "mean_weighted_J": statistics.fmean(a["weighted_J"] for _, a in paired),
        "mean_inequality": statistics.fmean(a["inequality"] for _, a in paired),
        "mean_weighted_KKT_p": statistics.fmean(a["weighted_KKT_p"] for _, a in paired),
        "mean_weighted_KKT_n": statistics.fmean(a["weighted_KKT_n"] for _, a in paired),
        "mean_weighted_R_stat": statistics.fmean(rstat),
        "median_weighted_R_stat": statistics.median(rstat),
        "max_weighted_R_stat": max(rstat),
        "mean_R_stat_ratio_adapted_over_control": statistics.fmean(ratio),
        "fraction_R_stat_improved": sum(value < 1.0 for value in ratio) / len(ratio),
        "fraction_R_stat_le_0p05": sum(value <= 0.05 for value in rstat) / len(rstat),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Frozen projected-n AWF weighted-objective pattern/size sensitivity"
    )
    parser.add_argument(
        "--instances-per-size", type=int, default=DEFAULT_INSTANCES_PER_SIZE
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    if args.instances_per_size <= 0:
        parser.error("--instances-per-size must be positive")
    return args


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    instances_per_size = int(args.instances_per_size)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    if checkpoint.get("n_update_mode") != "projected":
        raise RuntimeError("Expected projected-n checkpoint")
    table_gammas, tables = checkpoint_tables(checkpoint["model_state_dict"], device)
    token_gammas = logspace_gamma_grid(-10, 30, 32, device=str(device))
    model = MercuryFoundationSolver(
        table_gammas,
        tables,
        MODULATION_ID_ORDER,
        n_update_mode="projected",
    ).to(device=device, dtype=torch.float32)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.heads.ap_max = float(checkpoint["alpha_p_max"])
    model.heads.an_u_max = float(checkpoint["alpha_n_max"])
    model.heads.amu_max = float(checkpoint["alpha_mu_max"])
    model.eval()
    parameter_hash_before = model_tensor_hash(model)

    rows = []
    for m in SIZES:
        seed_all(INSTANCE_SEED + m)
        instances = [
            generate_feasible_instance(
                m=m,
                device=str(device),
                tables=tables,
                table_gammas=table_gammas,
                token_gammas=token_gammas,
                budget_mode="per_channel_fixed",
                constraint_structure="sparse",
                modulation_choices=["16QAM"],
            )
            for _ in range(instances_per_size)
        ]
        for instance_id, instance in enumerate(instances):
            pb = collate_variable([instance], device=str(device))
            with torch.no_grad():
                p_control, n_control, (_, mu_control) = model(
                    pb, T=T, warm_feas_steps=WARM_FEAS_STEPS
                )
            for pattern in PATTERNS:
                weights = make_weights(pattern, m, instance_id, device)
                with torch.no_grad():
                    p_adapted, n_adapted, (_, mu_adapted) = model(
                        pb,
                        T=T,
                        warm_feas_steps=WARM_FEAS_STEPS,
                        utility_weights=weights,
                    )
                for mode, p, n, mu in (
                    ("uniform_gradient_control", p_control, n_control, mu_control),
                    ("weighted_gradient_adaptation", p_adapted, n_adapted, mu_adapted),
                ):
                    metric = metrics_for_state(model, pb, p, n, mu, weights)
                    rows.append(
                        {
                            "m": m,
                            "instance_id": instance_id,
                            "K": int(instance["K"]),
                            "pattern": pattern,
                            "mode": mode,
                            "weight_min": float(weights.min().item()),
                            "weight_max": float(weights.max().item()),
                            "weight_mean": float(weights.mean().item()),
                            **metric,
                        }
                    )
        print(f"COMPLETED m={m}", flush=True)

    if model_tensor_hash(model) != parameter_hash_before:
        raise RuntimeError("Frozen checkpoint parameters changed")
    if not all(row["all_finite"] for row in rows):
        raise RuntimeError("Nonfinite output detected")

    summary_rows = []
    for m in SIZES:
        for pattern in PATTERNS:
            group = [row for row in rows if row["m"] == m and row["pattern"] == pattern]
            summary_rows.append({"m": m, "pattern": pattern, **summarize_group(group)})

    raw_path = output_dir / "weighted_size_pattern_raw.csv"
    with raw_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary_path = output_dir / "weighted_size_pattern_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    report = {
        "label": "zero-shot weighted-objective pattern and size sensitivity",
        "scope": "objective-family generalization within AWF; not arbitrary-task generalization",
        "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": sha256_file(CHECKPOINT),
        "checkpoint_frozen": True,
        "n_update_mode": "projected",
        "sizes": list(SIZES),
        "patterns": {
            "random_positive": "iid Uniform[0.5,1.5], normalized to arithmetic mean one",
            "grouped": "one contiguous half at 0.5 and the other at 1.5, orientation alternated by instance",
            "wide_two_level": "randomly permuted equal counts at 0.25 and 1.75",
        },
        "instances_per_size": instances_per_size,
        "instance_seed_rule": f"{INSTANCE_SEED} + m",
        "weight_seed_rule": f"{WEIGHT_SEED} + 10000*m + instance_id",
        "modulation": "16QAM",
        "constraint_type": "sparse",
        "T": T,
        "feasibility_warmup": WARM_FEAS_STEPS,
        "eta_eval": ETA_EVAL,
        "parameter_updates": 0,
        "hyperparameter_tuning": False,
        "predefined_paper_test_set_accessed": False,
        "summary": summary_rows,
        "raw_csv": str(raw_path),
        "summary_csv": str(summary_path),
    }
    report_path = output_dir / "weighted_size_pattern_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"REPORT={report_path}", flush=True)


if __name__ == "__main__":
    main()
