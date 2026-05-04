from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from .constants import MODULATION_ID_ORDER, TRAIN_MODULATION_CHOICES
from .data import (MercuryDataset, collate_variable, expand_curriculum_phases,
                   format_modulation_histogram, get_curriculum_phase,
                   make_fixed_validation_buckets, modulation_histogram)
from .evaluation import evaluate_fixed_buckets
from .losses import compute_loss
from .metrics import kkt_metrics
from .model import MercuryFoundationSolver


def schedule_training_objective(step: int, total_steps: int,
                                lam_ineq_final: float, lam_kktp_final: float,
                                stage2_frac: float = 0.55):
    frac = step / max(1, total_steps)
    stage2_start = max(1e-6, 1.0 - stage2_frac)

    if frac < stage2_start:
        local = frac / stage2_start
        lam_ineq = lam_ineq_final * (0.30 + 0.50 * min(1.0, local / 0.25))
        lam_kktp = max(1.2, 2.0 * lam_kktp_final) * (0.60 + 0.40 * min(1.0, local / 0.35))
        lam_kktn = 0.20 + 0.20 * local
        lam_imit_p = 0.10
        lam_imit_n = 0.00
        use_n_br_gap = False
        stage_name = 'stage1_kkt_bias'
    else:
        local = (frac - stage2_start) / max(1e-6, 1.0 - stage2_start)
        lam_ineq = 0.25 * lam_ineq_final
        lam_kktp = max(4.0, 6.0 * lam_kktp_final) * (0.95 + 0.05 * local)
        lam_kktn = 1.00 * (0.95 + 0.05 * local)
        lam_imit_p = 0.05
        lam_imit_n = 0.00
        use_n_br_gap = False
        stage_name = 'stage2_strong_p_kkt'

    return {
        'stage_name': stage_name,
        'lam_ineq': lam_ineq,
        'lam_kktp': lam_kktp,
        'lam_kktn': lam_kktn,
        'lam_imit_p': lam_imit_p,
        'lam_imit_n': lam_imit_n,
        'use_n_br_gap': use_n_br_gap,
    }


def train_model(device: str, tables, table_gammas, token_gammas, budget_mode: str = "per_channel_fixed",
                epochs: int = 8, steps_per_epoch: int = 200, batch_size: int = 16, lr: float = 1e-4,
                lam_ineq_final: float = 10.0, lam_kktp_final: float = 0.7, T_unroll: int = 64,
                br_train_steps_p: int = 300, br_train_steps_n: int = 300, p_active_tau: float = 5e-4,
                save_path: Optional[str] = None, save_best: bool = True,
                curriculum_phases: Optional[List[Dict]] = None,
                val_m_buckets: Optional[List[Tuple[int, int]]] = None,
                val_modulation_groups: Optional[Dict[str, List[str]]] = None,
                val_batches_per_bucket: int = 3, eval_every: int = 50,
                ema_beta: float = 0.90, stage2_frac: float = 0.55):
    total_steps = epochs * steps_per_epoch
    phases = expand_curriculum_phases(curriculum_phases, total_steps, default_m_min=32, default_m_max=512)

    ds = MercuryDataset(
        n_samples=epochs * steps_per_epoch * batch_size, m_min=32, m_max=512, device=device,
        tables=tables, table_gammas=table_gammas, token_gammas=token_gammas, budget_mode=budget_mode,
        m_buckets=phases[0]["m_buckets"], bucket_probs=phases[0].get("bucket_probs"),
        modulation_choices=phases[0].get("modulation_choices", TRAIN_MODULATION_CHOICES),
    )
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=lambda bl: collate_variable(bl, device=device))

    val_buckets = make_fixed_validation_buckets(
        device=device, tables=tables, table_gammas=table_gammas, token_gammas=token_gammas,
        budget_mode=budget_mode, batch_size=batch_size, n_batches=val_batches_per_bucket,
        m_buckets=val_m_buckets, modulation_groups=val_modulation_groups,
    )

    model = MercuryFoundationSolver(
        table_gammas=table_gammas,
        tables=tables,
        modulation_id_order=MODULATION_ID_ORDER,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    step_global = 0
    best_score = float('inf')
    ema = {}
    current_phase_name = None
    current_objective_stage = None

    it = iter(dl)
    model.train()
    for ep in range(1, epochs + 1):
        for st in range(1, steps_per_epoch + 1):
            step_global += 1
            phase = get_curriculum_phase(phases, step_global, total_steps)
            if current_phase_name != phase.get("name"):
                current_phase_name = phase.get("name")
                ds.set_sampling_scheme(phase["m_buckets"], phase.get("bucket_probs"))
                ds.modulation_choices = phase.get("modulation_choices", TRAIN_MODULATION_CHOICES)
                print(f"[curriculum] step={step_global:04d} phase={current_phase_name} buckets={phase['m_buckets']} mods={ds.modulation_choices}")

            try:
                pb = next(it)
            except StopIteration:
                it = iter(dl)
                pb = next(it)

            p, n, (_, mu) = model(pb, T=T_unroll)
            obj_cfg = schedule_training_objective(step_global, total_steps, lam_ineq_final, lam_kktp_final, stage2_frac=stage2_frac)
            if current_objective_stage != obj_cfg["stage_name"]:
                current_objective_stage = obj_cfg["stage_name"]
                print(
                    f"[objective] step={step_global:04d} stage={current_objective_stage} "
                    f"lam_ineq={obj_cfg['lam_ineq']:.2f} lam_kktp={obj_cfg['lam_kktp']:.2f} "
                    f"lam_kktn={obj_cfg['lam_kktn']:.2f} lam_imit_p={obj_cfg['lam_imit_p']:.2f} "
                    f"lam_imit_n={obj_cfg['lam_imit_n']:.2f} use_n_gap={obj_cfg['use_n_br_gap']}"
                )
            loss, met = compute_loss(
                model, pb, p, n, mu,
                obj_cfg['lam_ineq'], obj_cfg['lam_kktp'], obj_cfg['lam_kktn'],
                p_active_tau, br_train_steps_p, br_train_steps_n,
                lam_imit_p=obj_cfg['lam_imit_p'], lam_imit_n=obj_cfg['lam_imit_n'],
                use_n_br_gap=obj_cfg['use_n_br_gap'],
            )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            weighted_terms = {
                "loss": met["loss"],
                "gap": met["gap"],
                "gapLoss": met["gapLoss"],
                "ineq_term": obj_cfg['lam_ineq'] * met["ineqSharp"],
                "kktp_term": obj_cfg['lam_kktp'] * met["kktpReg"],
                "kktn_term": obj_cfg['lam_kktn'] * met["kktnReg"],
                "imitp_term": obj_cfg['lam_imit_p'] * met["imitP"],
                "imitn_term": obj_cfg['lam_imit_n'] * met["imitN"],
            }
            for k, v in weighted_terms.items():
                ema[k] = v if k not in ema else (ema_beta * ema[k] + (1.0 - ema_beta) * v)

            mod_stats = modulation_histogram(pb.mod_id)

            if st % 25 == 0:
                km = kkt_metrics(model, pb, p, n, None, mu, p_active_tau)
                print(
                    f"[train ep={ep:02d} step={st:04d}] stage={current_objective_stage} "
                    f"loss={met['loss']:.4f} ema={ema['loss']:.4f} gap={met['gap']:.4f} gapL={met['gapLoss']:.4f} "
                    f"ineq={met['ineq']:.2e} ineqSharp={met['ineqSharp']:.2e} ineqW={weighted_terms['ineq_term']:.3f} "
                    f"kktpW={weighted_terms['kktp_term']:.3f} kktnW={weighted_terms['kktn_term']:.3f} "
                    f"imitPW={weighted_terms['imitp_term']:.3f} imitNW={weighted_terms['imitn_term']:.3f} "
                    f"mods=({format_modulation_histogram(mod_stats)}) "
                    f"Jmean={met['Jmean']:.4f} kkt_p_act={km['kkt_p_active']:.2e} "
                    f"kkt_n={km['kkt_n']:.2e} (act={km['kkt_n_active']:.2e}, ineq={km['kkt_n_ineq']:.2e}) "
                    f"|sumP|={km['sumP_abs']:.2e} |sumN|={km['sumN_abs']:.2e} "
                    f"muMax={km['mu_max']:.2e} nuStarMax={km['nu_max']:.2e}"
                )

            if (step_global % eval_every == 0) or (ep == epochs and st == steps_per_epoch):
                val_out = evaluate_fixed_buckets(
                    model, val_buckets, T_unroll=T_unroll, p_active_tau=p_active_tau,
                    br_eval_steps_p=max(80, br_train_steps_p // 2),
                    br_eval_steps_n=max(80, br_train_steps_n // 2),
                    lam_ineq=lam_ineq_final, lam_kktp=lam_kktp_final, lam_kktn=0.05,
                )
                vs = val_out["summary"]
                print(
                    f"[val step={step_global:04d}] score={vs['score']:.4f} avg_loss={vs['loss']:.4f} "
                    f"worst_loss={vs['worst_bucket_loss']:.4f} ineq={vs['ineq']:.2e} kkt_p={vs['kkt_p_active']:.2e} kkt_n={vs['kkt_n']:.2e}"
                )
                for bucket_name, bucket_met in val_out["buckets"].items():
                    print(
                        f"    [{bucket_name}] loss={bucket_met['loss']:.4f} gap={bucket_met['gap']:.4f} "
                        f"ineq={bucket_met['ineq']:.2e} kkt_p={bucket_met['kkt_p_active']:.2e} kkt_n={bucket_met['kkt_n']:.2e}"
                    )

                if save_path is not None and save_best:
                    cur_score = float(vs['score'])
                    if cur_score < best_score:
                        best_score = cur_score
                        torch.save({
                            "model_state_dict": model.state_dict(),
                            "val_summary": vs,
                            "curriculum_phase": current_phase_name,
                            "step_global": step_global,
                        }, save_path)
                        print(f"[checkpoint] saved best to {save_path} (val_score={best_score:.6f})")
    return model
