"""Train a projected-n AWF model with the public release protocol."""

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mercury.checkpoint import checkpoint_tables, extract_state_dict
from mercury.constants import TRAIN_MODULATION_CHOICES
from mercury.training import train_model
from mercury.utils import logspace_gamma_grid, seed_all


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Train the public projected-n AWF model")
    parser.add_argument(
        "--table-source",
        default=str(ROOT / "checkpoints" / "awf_projected_n_seed123_final.pt"),
        help="Checkpoint supplying the fixed MMSE/I lookup tables only",
    )
    parser.add_argument("--output", default=str(ROOT / "outputs" / "awf_projected_n_final.pt"))
    parser.add_argument("--report", default=str(ROOT / "outputs" / "training_report.json"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--steps-per-epoch", type=int, default=50)
    args = parser.parse_args()

    device = torch.device(args.device)
    source_path = Path(args.table_source).resolve()
    output_path = Path(args.output).resolve()
    report_path = Path(args.report).resolve()
    for path in (output_path, report_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
        path.parent.mkdir(parents=True, exist_ok=True)

    source = torch.load(source_path, map_location="cpu", weights_only=False)
    source_state = extract_state_dict(source)
    table_gammas, tables = checkpoint_tables(source_state, device)
    token_gammas = logspace_gamma_grid(-10, 30, 32, device=str(device))
    curriculum = [
        {
            "name": "warmup_small_mixed_mod",
            "until_frac": 0.30,
            "m_buckets": [(32, 96)],
            "bucket_probs": [1.0],
            "modulation_choices": TRAIN_MODULATION_CHOICES,
        },
        {
            "name": "mid_mixed",
            "until_frac": 0.65,
            "m_buckets": [(32, 96), (128, 256)],
            "bucket_probs": [0.5, 0.5],
            "modulation_choices": TRAIN_MODULATION_CHOICES,
        },
        {
            "name": "full_range",
            "until_frac": 1.0,
            "m_buckets": [(32, 96), (128, 256), (384, 512)],
            "bucket_probs": [0.34, 0.33, 0.33],
            "modulation_choices": TRAIN_MODULATION_CHOICES,
        },
    ]

    seed_all(args.seed)
    started = datetime.now(timezone.utc)
    model = train_model(
        str(device),
        tables,
        table_gammas,
        token_gammas,
        budget_mode="per_channel_fixed",
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        batch_size=16,
        lr=1e-4,
        lam_ineq_final=10.0,
        lam_kktp_final=0.7,
        T_unroll=64,
        br_train_steps_p=300,
        br_train_steps_n=300,
        p_active_tau=5e-4,
        save_path=None,
        save_best=False,
        curriculum_phases=curriculum,
        val_m_buckets=[(32, 64), (96, 160), (224, 320), (384, 512)],
        val_modulation_groups={
            "mod16": ["16QAM"],
            "mod64": ["64QAM"],
            "mixed": ["16QAM", "64QAM"],
        },
        val_batches_per_bucket=2,
        eval_every=max(1, args.epochs * args.steps_per_epoch),
        ema_beta=0.90,
        stage2_frac=0.55,
        alpha_p_max=0.08,
        n_update_mode="projected",
    )
    finished = datetime.now(timezone.utc)
    if model.n_update_mode != "projected":
        raise RuntimeError("Training did not use the projected n update")

    checkpoint = {
        "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "n_update_mode": "projected",
        "alpha_p_max": 0.08,
        "alpha_n_max": 0.6,
        "alpha_mu_max": 0.9,
        "training_seed": args.seed,
        "T": 64,
        "feasibility_warmup": 16,
        "table_source_sha256": sha256_file(source_path),
    }
    torch.save(checkpoint, output_path)
    report = {
        "checkpoint": str(output_path),
        "checkpoint_sha256": sha256_file(output_path),
        "n_update_mode": "projected",
        "seed": args.seed,
        "epochs": args.epochs,
        "steps_per_epoch": args.steps_per_epoch,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "started_at_utc": started.isoformat(),
        "finished_at_utc": finished.isoformat(),
        "test_set_accessed": False,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
