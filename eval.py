import argparse
from pathlib import Path

import torch

from mercury.constants import MODULATION_ID_ORDER, TEST_MODULATION_GROUPS
from mercury.evaluation import evaluate_constraint_structure_generalization, evaluate_modulation_generalization
from mercury.model import MercuryFoundationSolver
from mercury.qam import build_tables
from mercury.utils import logspace_gamma_grid, seed_all


def load_model(checkpoint_path: str, device: str):
    table_gammas = logspace_gamma_grid(-10, 30, 128, device=device)
    tables = build_tables(device=device, gammas=table_gammas, n_samples_per_gamma=15000)
    model = MercuryFoundationSolver(
        table_gammas=table_gammas,
        tables=tables,
        modulation_id_order=MODULATION_ID_ORDER
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model, tables, table_gammas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    seed_all(123)
    device = args.device
    token_gammas = logspace_gamma_grid(-10, 30, 32, device=device)
    model, tables, table_gammas = load_model(args.checkpoint, device)

    modulation_results = evaluate_modulation_generalization(
        model, device, m_list=[32, 128, 512, 1024],
        tables=tables, table_gammas=table_gammas, token_gammas=token_gammas,
        budget_mode="per_channel_fixed", modulation_groups=TEST_MODULATION_GROUPS,
        n_batches=2, batch_size=4, T_unroll=64,
    )
    print("modulation_results:")
    print(modulation_results)

    constraint_results = evaluate_constraint_structure_generalization(
        model, device, m_list=[128, 512, 1024],
        tables=tables, table_gammas=table_gammas, token_gammas=token_gammas,
        budget_mode="per_channel_fixed", structures=("sparse", "group", "prefix", "dense"),
        n_batches=2, batch_size=4, T_unroll=64,
    )
    print("constraint_results:")
    print(constraint_results)


if __name__ == "__main__":
    main()
