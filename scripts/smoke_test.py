import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mercury.checkpoint import load_awf_checkpoint
from mercury.data import collate_variable, generate_feasible_instance
from mercury.metrics import Jmean, inequality_violation, kkt_metrics, projected_stationarity_residual
from mercury.utils import logspace_gamma_grid, seed_all


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="AWF checkpoint smoke test (no baseline)")
    parser.add_argument(
        "--checkpoint",
        default=str(ROOT / "checkpoints" / "awf_projected_n_seed123_final.pt"),
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--instances", type=int, default=3)
    parser.add_argument("--m", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--output", default=str(ROOT / "outputs" / "smoke_test.json"))
    args = parser.parse_args()

    device = torch.device(args.device)
    checkpoint_path = Path(args.checkpoint).resolve()
    model, checkpoint, tables, table_gammas = load_awf_checkpoint(checkpoint_path, device)
    token_gammas = logspace_gamma_grid(-10, 30, 32, device=str(device))
    seed_all(args.seed)
    rows = []
    for instance_id in range(args.instances):
        instance = generate_feasible_instance(
            m=args.m,
            device=str(device),
            tables=tables,
            table_gammas=table_gammas,
            token_gammas=token_gammas,
            budget_mode="per_channel_fixed",
            constraint_structure="sparse",
            modulation_choices=["16QAM"],
        )
        pb = collate_variable([instance], device=str(device))
        with torch.no_grad():
            p, n, (_, mu) = model(pb, T=64, warm_feas_steps=16)
            kkt = kkt_metrics(model, pb, p, n, None, mu)
            rows.append(
                {
                    "instance_id": instance_id,
                    "m": args.m,
                    "K": int(instance["K"]),
                    "J": float(Jmean(model, pb, p, n).item()),
                    "inequality": float(inequality_violation(pb, p).item()),
                    "KKT_p": float(kkt["kkt_p_active"]),
                    "KKT_n": float(kkt["kkt_n"]),
                    "R_stat": float(projected_stationarity_residual(model, pb, p, n, mu).item()),
                    "sum_p_error": float((p.sum(-1, keepdim=True) - pb.P).abs().max().item()),
                    "sum_n_error": float((n.sum(-1, keepdim=True) - pb.N).abs().max().item()),
                    "all_finite": bool(all(torch.isfinite(x).all().item() for x in (p, n, mu))),
                }
            )

    report = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "n_update_mode": checkpoint.get("n_update_mode"),
        "seed": args.seed,
        "T": 64,
        "feasibility_warmup": 16,
        "eta_eval": 0.01,
        "baseline_used": False,
        "instances": rows,
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"REPORT={output}")


if __name__ == "__main__":
    main()
