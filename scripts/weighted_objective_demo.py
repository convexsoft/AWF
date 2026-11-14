import csv
import hashlib
import json
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
OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "weighted_utility_toy_seed20260916"
)
INSTANCE_SEED = 20260916
WEIGHT_SEED = 20260917
N_INSTANCES = 3
M = 64
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


def fixed_two_level_weights(instance_id: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(WEIGHT_SEED + instance_id)
    weights = torch.cat(
        [
            torch.full((M // 2,), 0.5, dtype=torch.float32),
            torch.full((M - M // 2,), 1.5, dtype=torch.float32),
        ]
    )
    weights = weights[torch.randperm(M, generator=generator)]
    return weights.view(1, M).to(device)


def metric_row(model, pb, p, n, mu, weights, instance_id: int, mode: str):
    kkt = kkt_metrics(
        model,
        pb,
        p,
        n,
        None,
        mu,
        utility_weights=weights,
    )
    return {
        "instance_id": instance_id,
        "mode": mode,
        "m": M,
        "K": int(pb.mask_k.sum().item()),
        "modulation": "16QAM",
        "constraint_type": "sparse",
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


def main() -> None:
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    if checkpoint.get("n_update_mode") != "projected":
        raise RuntimeError("The toy study requires the projected-n checkpoint")
    table_gammas, tables = checkpoint_tables(checkpoint["model_state_dict"], device)
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

    token_gammas = logspace_gamma_grid(-10, 30, 32, device=str(device))
    seed_all(INSTANCE_SEED)
    rows = []
    paired = []
    for instance_id in range(N_INSTANCES):
        instance = generate_feasible_instance(
            m=M,
            device=str(device),
            tables=tables,
            table_gammas=table_gammas,
            token_gammas=token_gammas,
            budget_mode="per_channel_fixed",
            constraint_structure="sparse",
            modulation_choices=["16QAM"],
        )
        pb = collate_variable([instance], device=str(device))
        weights = fixed_two_level_weights(instance_id, device)
        if not torch.allclose(weights.mean(), torch.tensor(1.0, device=device)):
            raise RuntimeError("The fixed utility weights must have unit mean")

        with torch.no_grad():
            p_control, n_control, (_, mu_control) = model(
                pb,
                T=T,
                warm_feas_steps=WARM_FEAS_STEPS,
            )
            p_adapted, n_adapted, (_, mu_adapted) = model(
                pb,
                T=T,
                warm_feas_steps=WARM_FEAS_STEPS,
                utility_weights=weights,
            )

        control = metric_row(
            model, pb, p_control, n_control, mu_control, weights, instance_id,
            "uniform_gradient_control",
        )
        adapted = metric_row(
            model, pb, p_adapted, n_adapted, mu_adapted, weights, instance_id,
            "weighted_gradient_adaptation",
        )
        rows.extend([control, adapted])
        paired.append(
            {
                "instance_id": instance_id,
                "weighted_R_stat_control": control["weighted_R_stat"],
                "weighted_R_stat_adapted": adapted["weighted_R_stat"],
                "R_stat_ratio_adapted_over_control": (
                    adapted["weighted_R_stat"] / control["weighted_R_stat"]
                ),
                "weighted_J_control": control["weighted_J"],
                "weighted_J_adapted": adapted["weighted_J"],
                "weighted_J_change": adapted["weighted_J"] - control["weighted_J"],
                "p_L1_change": float((p_adapted - p_control).abs().sum().item()),
                "n_L1_change": float((n_adapted - n_control).abs().sum().item()),
            }
        )

    parameter_hash_after = model_tensor_hash(model)
    if parameter_hash_after != parameter_hash_before:
        raise RuntimeError("Frozen model parameters changed during the toy evaluation")
    if not all(row["all_finite"] for row in rows):
        raise RuntimeError("Nonfinite output detected")

    modes = ("uniform_gradient_control", "weighted_gradient_adaptation")
    aggregate = {}
    for mode in modes:
        selected = [row for row in rows if row["mode"] == mode]
        aggregate[mode] = {
            metric: sum(row[metric] for row in selected) / len(selected)
            for metric in (
                "weighted_J",
                "inequality",
                "weighted_KKT_p",
                "weighted_KKT_n",
                "weighted_R_stat",
            )
        }

    raw_csv = OUTPUT_DIR / "weighted_utility_toy_raw.csv"
    with raw_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "label": "toy zero-shot weighted-utility adaptation of the frozen projected-n AWF solver",
        "scope": "illustrative toy example; not a broad foundation-model claim",
        "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": sha256_file(CHECKPOINT),
        "checkpoint_frozen": parameter_hash_before == parameter_hash_after,
        "n_update_mode": "projected",
        "instance_seed": INSTANCE_SEED,
        "weight_seed": WEIGHT_SEED,
        "instances": N_INSTANCES,
        "m": M,
        "modulation": "16QAM",
        "constraint_type": "sparse",
        "T": T,
        "feasibility_warmup": WARM_FEAS_STEPS,
        "eta_eval": ETA_EVAL,
        "weight_definition": "32 entries equal to 0.5 and 32 entries equal to 1.5, independently fixed-permuted per instance; arithmetic mean 1",
        "adaptation": "only multiply the analytical p- and n-gradient fields by w_i; no parameter update",
        "baseline_used": False,
        "test_set_accessed": False,
        "aggregate": aggregate,
        "paired": paired,
        "raw_rows": rows,
    }
    report_json = OUTPUT_DIR / "weighted_utility_toy_report.json"
    report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps({"aggregate": aggregate, "paired": paired}, indent=2))
    print(f"RAW_CSV={raw_csv}")
    print(f"REPORT_JSON={report_json}")


if __name__ == "__main__":
    main()
