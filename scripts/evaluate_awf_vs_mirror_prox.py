"""Evaluate projected-n AWF and projected Mirror-Prox on paired instances."""

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mercury.checkpoint import load_awf_checkpoint
from mercury.data import collate_variable, generate_feasible_instance
from mercury.metrics import Jmean, inequality_violation, kkt_metrics, projected_stationarity_residual
from mercury.optimization import mirror_prox_extragradient_primal_dual_stable
from mercury.utils import logspace_gamma_grid, seed_all


def weights_for(pattern, m, instance_id, device, seed):
    if pattern == "none":
        return None
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 10000 * m + instance_id)
    if pattern == "random_positive":
        w = 0.5 + torch.rand(m, generator=generator)
        w = w / w.mean()
    elif pattern == "grouped":
        w = torch.cat([torch.full((m // 2,), 0.5), torch.full((m - m // 2,), 1.5)])
        if instance_id % 2:
            w = torch.flip(w, dims=(0,))
    elif pattern == "wide_two_level":
        w = torch.cat([torch.full((m // 2,), 0.25), torch.full((m - m // 2,), 1.75)])
        w = w[torch.randperm(m, generator=generator)]
    else:
        raise ValueError(pattern)
    return w.view(1, m).to(device=device, dtype=torch.float32)


@torch.no_grad()
def quality(model, pb, p, n, mu, weights):
    kkt = kkt_metrics(model, pb, p, n, None, mu, utility_weights=weights)
    return {
        "J": float(Jmean(model, pb, p, n, utility_weights=weights).item()),
        "inequality": float(inequality_violation(pb, p).item()),
        "KKT_p": float(kkt["kkt_p_active"]),
        "KKT_n": float(kkt["kkt_n"]),
        "R_stat": float(
            projected_stationarity_residual(
                model, pb, p, n, mu, stat_step=0.01, utility_weights=weights
            ).item()
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default=str(ROOT / "checkpoints" / "awf_projected_n_seed123_final.pt"),
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--sizes", nargs="+", type=int, default=[32, 64, 128, 256, 512, 1024])
    parser.add_argument("--instances", type=int, default=3)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--modulation", choices=["16QAM", "64QAM", "256QAM"], default="16QAM")
    parser.add_argument("--constraint", choices=["sparse", "group", "prefix", "dense"], default="sparse")
    parser.add_argument(
        "--weight-pattern",
        choices=["none", "random_positive", "grouped", "wide_two_level"],
        default="none",
    )
    parser.add_argument("--mp-iterations", type=int, default=3000)
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "awf_vs_mirror_prox"))
    args = parser.parse_args()
    if args.instances <= 0 or args.mp_iterations <= 0:
        parser.error("instances and mp-iterations must be positive")

    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    output_dir.mkdir(parents=True)
    device = torch.device(args.device)
    model, checkpoint, tables, table_gammas = load_awf_checkpoint(args.checkpoint, device)
    token_gammas = logspace_gamma_grid(-10, 30, 32, device=str(device))

    rows = []
    for m in args.sizes:
        seed_all(args.seed + m)
        for instance_id in range(args.instances):
            instance = generate_feasible_instance(
                m=m,
                device=str(device),
                tables=tables,
                table_gammas=table_gammas,
                token_gammas=token_gammas,
                budget_mode="per_channel_fixed",
                constraint_structure=args.constraint,
                modulation_choices=[args.modulation],
            )
            pb = collate_variable([instance], device=str(device))
            weights = weights_for(args.weight_pattern, m, instance_id, device, args.seed + 100000)
            with torch.no_grad():
                p_awf, n_awf, (_, mu_awf) = model(
                    pb, T=64, warm_feas_steps=16, utility_weights=weights
                )
                p_mp, n_mp, (_, mu_mp), iterations = mirror_prox_extragradient_primal_dual_stable(
                    model,
                    pb,
                    T=args.mp_iterations,
                    step_p=0.01,
                    step_n=0.03,
                    step_mu=0.05,
                    warm_feas_steps=80,
                    early_stop=False,
                    utility_weights=weights,
                )
            for method, p, n, mu in (
                ("Projected-AWF", p_awf, n_awf, mu_awf),
                ("Mirror-Prox", p_mp, n_mp, mu_mp),
            ):
                rows.append(
                    {
                        "m": m,
                        "instance_id": instance_id,
                        "K": int(instance["K"]),
                        "modulation": args.modulation,
                        "constraint": args.constraint,
                        "weight_pattern": args.weight_pattern,
                        "method": method,
                        "iterations": 64 if method == "Projected-AWF" else iterations,
                        **quality(model, pb, p, n, mu, weights),
                    }
                )
        print(f"COMPLETED m={m}", flush=True)

    raw_path = output_dir / "per_instance.csv"
    with raw_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary = []
    for m in args.sizes:
        for method in ("Projected-AWF", "Mirror-Prox"):
            selected = [row for row in rows if row["m"] == m and row["method"] == method]
            summary.append(
                {
                    "m": m,
                    "method": method,
                    **{
                        metric: statistics.fmean(row[metric] for row in selected)
                        for metric in ("J", "inequality", "KKT_p", "KKT_n", "R_stat")
                    },
                }
            )
    report = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "n_update_mode": checkpoint["n_update_mode"],
        "eta_eval": 0.01,
        "awf": {"T": 64, "warm_feas_steps": 16},
        "mirror_prox": {
            "iterations": args.mp_iterations,
            "alpha_p": 0.01,
            "alpha_n": 0.03,
            "alpha_mu": 0.05,
            "warm_feas_steps": 80,
        },
        "summary": summary,
        "raw_csv": str(raw_path),
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
