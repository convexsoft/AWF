import time

import torch

from mercury.constants import TEST_MODULATION_GROUPS, TRAIN_MODULATION_CHOICES
from mercury.evaluation import (
    eval_model_with_baseline,
    evaluate_constraint_structure_generalization,
    evaluate_modulation_generalization,
    run_br_usefulness_check,
)
from mercury.qam import build_tables
from mercury.training import train_model
from mercury.utils import logspace_gamma_grid, seed_all


def main():
    seed_all(123)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    budget_mode = "per_channel_fixed"
    table_gammas = logspace_gamma_grid(-10, 30, 128, device=device)
    token_gammas = logspace_gamma_grid(-10, 30, 32, device=device)

    print("Simulating MMSE/I tables...")
    tables = build_tables(device=device, gammas=table_gammas, n_samples_per_gamma=15000)
    print(f"Train modulations: {TRAIN_MODULATION_CHOICES} | Test-only held-out modulation: {['256QAM']}")

    save_name = f"awf_model_{int(time.time())}.pt"
    curriculum_phases = [
        {"name": "warmup_small_mixed_mod", "until_frac": 0.30, "m_buckets": [(32, 96)], "bucket_probs": [1.0], "modulation_choices": TRAIN_MODULATION_CHOICES},
        {"name": "mid_mixed", "until_frac": 0.65, "m_buckets": [(32, 96), (128, 256)], "bucket_probs": [0.5, 0.5], "modulation_choices": TRAIN_MODULATION_CHOICES},
        {"name": "full_range", "until_frac": 1.00, "m_buckets": [(32, 96), (128, 256), (384, 512)], "bucket_probs": [0.34, 0.33, 0.33], "modulation_choices": TRAIN_MODULATION_CHOICES},
    ]

    model = train_model(
        device,
        tables,
        table_gammas,
        token_gammas,
        budget_mode,
        epochs=1,#4
        steps_per_epoch=100,#200
        batch_size=16,
        lr=1e-4,
        save_path=save_name,
        save_best=True,
        curriculum_phases=curriculum_phases,
        val_m_buckets=[(32, 64), (96, 160), (224, 320), (384, 512)],
        val_modulation_groups={"mod16": ["16QAM"], "mod64": ["64QAM"], "mixed": ["16QAM", "64QAM"]},
        val_batches_per_bucket=2,
        eval_every=200,
    )
    final_name = save_name.replace('.pt', '_final.pt')
    torch.save(model.state_dict(), final_name)
    print(f"[checkpoint] saved final to {final_name}")

    print("\n=== BR usefulness check ===")
    run_br_usefulness_check(
        model,
        device,
        tables,
        table_gammas,
        token_gammas,
        budget_mode,
        m_list=[16, 32, 64, 128, 256, 512, 1024],
        batch_size=8,
        br_steps_p=300,
        br_steps_n=500,
    )

    print("\n=== Baseline check with STABLE MirrorProx ===")
    for m in [16, 32, 64, 128, 256, 512, 1024]:
        out = eval_model_with_baseline(model, device, m, tables, table_gammas, token_gammas, budget_mode)
        L, B = out["learned"], out["mirrorprox"]
        Ln = L.get("kkt_n", float("nan"))
        Bn = B.get("kkt_n", float("nan"))
        print(
            f"[m={m:4d}] Learned     J={L['J']:.4f} ineq={L['ineq']:.2e} "
            f"kkt_act={L['kkt_p_active']:.2e} kkt_n={Ln:.2e} runtime={L['runtime_ms']:.1f}ms"
        )
        print(
            f"         MirrorProx J={B['J']:.4f} ineq={B['ineq']:.2e} "
            f"kkt_act={B['kkt_p_active']:.2e} kkt_n={Bn:.2e} runtime={B['runtime_ms']:.1f}ms"
        )

    print("\n=== Modulation-format generalization check ===")
    modulation_results = evaluate_modulation_generalization(
        model,
        device,
        m_list=[32, 128, 512, 1024],
        tables=tables,
        table_gammas=table_gammas,
        token_gammas=token_gammas,
        budget_mode=budget_mode,
        modulation_groups=TEST_MODULATION_GROUPS,
        n_batches=2,
        batch_size=4,
        T_unroll=64,
        constraint_structure="sparse",
    )
    for mod_group, per_m in modulation_results.items():
        print(f"[modulation={mod_group}]")
        for m, out in per_m.items():
            L, B = out["learned"], out["mirrorprox"]
            print(
                f"  m={m:4d} | L: J={L['J']:.4f}, ineq={L['ineq']:.2e}, kkt_p={L['kkt_p_active']:.2e}, kkt_n={L['kkt_n']:.2e}, "
                f"t={L['runtime_ms']:.1f}ms || MP: J={B['J']:.4f}, ineq={B['ineq']:.2e}, "
                f"kkt_p={B['kkt_p_active']:.2e}, kkt_n={B['kkt_n']:.2e}, t={B['runtime_ms']:.1f}ms"
            )

    print("\n=== Constraint-structure generalization check ===")
    constraint_results = evaluate_constraint_structure_generalization(
        model,
        device,
        m_list=[128, 512, 1024],
        tables=tables,
        table_gammas=table_gammas,
        token_gammas=token_gammas,
        budget_mode=budget_mode,
        structures=("sparse", "group", "prefix", "dense"),
        n_batches=2,
        batch_size=4,
        T_unroll=64,
    )
    for structure, per_m in constraint_results.items():
        print(f"[structure={structure}]")
        for m, out in per_m.items():
            L, B = out["learned"], out["mirrorprox"]
            print(
                f"  m={m:4d} | L: J={L['J']:.4f}, ineq={L['ineq']:.2e}, kkt_p={L['kkt_p_active']:.2e}, kkt_n={L['kkt_n']:.2e}, "
                f"t={L['runtime_ms']:.1f}ms || MP: J={B['J']:.4f}, ineq={B['ineq']:.2e}, "
                f"kkt_p={B['kkt_p_active']:.2e}, kkt_n={B['kkt_n']:.2e}, t={B['runtime_ms']:.1f}ms"
            )


if __name__ == "__main__":
    main()
