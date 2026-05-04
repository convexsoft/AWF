import time
from typing import Dict, List, Optional

import torch

from .constants import TEST_MODULATION_GROUPS
from .data import ProblemBatch, collate_variable, generate_feasible_instance
from .losses import compute_loss
from .metrics import Jmean, kkt_metrics
from .optimization import br_n_u_pg, br_p_projected_primal_dual, mirror_prox_extragradient_primal_dual_stable


@torch.no_grad()
def evaluate_fixed_buckets(model, val_buckets, T_unroll: int = 64, p_active_tau: float = 5e-4,
                           br_eval_steps_p: int = 150, br_eval_steps_n: int = 150,
                           lam_ineq: float = 10.0, lam_kktp: float = 0.7, lam_kktn: float = 0.05):
    model.eval()
    per_bucket = {}
    for name, batches in val_buckets.items():
        rows = []
        for pb in batches:
            p, n, (_, mu) = model(pb, T=T_unroll)
            _, met = compute_loss(
                model, pb, p, n, mu,
                lam_ineq=lam_ineq, lam_kktp=lam_kktp, lam_kktn=lam_kktn,
                p_active_tau=p_active_tau, br_steps_p=br_eval_steps_p, br_steps_n=br_eval_steps_n,
            )
            km = kkt_metrics(model, pb, p, n, None, mu, p_active_tau)
            row = dict(met)
            row.update({
                "kkt_p_active": float(km["kkt_p_active"]),
                "kkt_n": float(km["kkt_n"]),
                "kkt_n_active": float(km["kkt_n_active"]),
                "kkt_n_ineq": float(km["kkt_n_ineq"]),
                "sumP_abs": float(km["sumP_abs"]),
                "sumN_abs": float(km["sumN_abs"]),
                "mu_max": float(km["mu_max"]),
                "nu_max": float(km["nu_max"]),
            })
            rows.append(row)

        agg = {}
        keys = rows[0].keys()
        for k in keys:
            agg[k] = sum(r[k] for r in rows) / len(rows)
        per_bucket[name] = agg

    summary_keys = ["loss", "gap", "gapLoss", "ineq", "Jmean", "kktpReg", "kktnReg", "kkt_p_active", "kkt_n"]
    summary = {k: sum(per_bucket[b][k] for b in per_bucket) / len(per_bucket) for k in summary_keys}
    summary["worst_bucket_loss"] = max(per_bucket[b]["loss"] for b in per_bucket)
    summary["score"] = 0.7 * summary["loss"] + 0.3 * summary["worst_bucket_loss"]
    model.train()
    return {"summary": summary, "buckets": per_bucket}


@torch.no_grad()
def validate_br_quality(model, pb: ProblemBatch, br_steps_p: int = 300, br_steps_n: int = 500, p_active_tau: float = 5e-4):
    p0, n0, (_, mu0) = model(pb, T=34)
    J0 = Jmean(model, pb, p0, n0)

    p_br = br_p_projected_primal_dual(model, pb, n0, steps=br_steps_p, lr_p=0.05, lr_mu=0.25)
    n_br = br_n_u_pg(model, pb, p0, steps=br_steps_n, lr_u=0.15, n_init=n0)

    J_pbr = Jmean(model, pb, p_br, n0)
    J_nbr = Jmean(model, pb, p0, n_br)

    km0 = kkt_metrics(model, pb, p0, n0, None, mu0, p_active_tau)
    Ap_br = (pb.A @ p_br.unsqueeze(-1)).squeeze(-1) * pb.mask_k.to(torch.float32)
    mu_br = (Ap_br - pb.p_hat).clamp_min(0.0)
    km_pbr = kkt_metrics(model, pb, p_br, n0, None, mu_br, p_active_tau)
    km_nbr = kkt_metrics(model, pb, p0, n_br, None, mu0, p_active_tau)

    return {
        "p_improve_mean": float((J_pbr - J0).mean().item()),
        "p_improve_hit_rate": float(((J_pbr - J0) > 1e-10).float().mean().item()),
        "n_improve_mean": float((J0 - J_nbr).mean().item()),
        "n_improve_hit_rate": float(((J0 - J_nbr) > 1e-10).float().mean().item()),
        "kkt_p_before": float(km0["kkt_p_active"]),
        "kkt_p_after_pbr": float(km_pbr["kkt_p_active"]),
        "kkt_n_before": float(km0["kkt_n"]),
        "kkt_n_after_nbr": float(km_nbr["kkt_n"]),
        "ineq_before": float(km0["ineq"]),
        "ineq_after_pbr": float(km_pbr["ineq"]),
    }


@torch.no_grad()
def run_br_usefulness_check(model, device, tables, table_gammas, token_gammas, budget_mode,
                            m_list=(16, 32, 128), batch_size: int = 8, br_steps_p: int = 300, br_steps_n: int = 500,
                            constraint_structure: str = "sparse"):
    model.eval()
    for m in m_list:
        instances = [
            generate_feasible_instance(
                m, device, tables, table_gammas, token_gammas, budget_mode,
                constraint_structure=constraint_structure,
            )
            for _ in range(batch_size)
        ]
        pb = collate_variable(instances, device=device)
        out = validate_br_quality(model, pb, br_steps_p=br_steps_p, br_steps_n=br_steps_n)
        print(
            f"[m={m:4d}] "
            f"p_improve={out['p_improve_mean']:+.4e} hit={100.0*out['p_improve_hit_rate']:.2f}% "
            f"n_improve={out['n_improve_mean']:+.4e} hit={100.0*out['n_improve_hit_rate']:.2f}% "
            f"kkt_p {out['kkt_p_before']:.2e}->{out['kkt_p_after_pbr']:.2e} "
            f"kkt_n {out['kkt_n_before']:.2e}->{out['kkt_n_after_nbr']:.2e} "
            f"ineq {out['ineq_before']:.2e}->{out['ineq_after_pbr']:.2e}"
        )


@torch.no_grad()
def eval_model_with_baseline(model, device, m, tables, table_gammas, token_gammas, budget_mode,
                             n_batches: int = 4, batch_size: int = 6, T_unroll: int = 64, p_active_tau: float = 5e-4,
                             constraint_structure: str = "sparse", modulation_choices: Optional[List[str]] = None):
    model.eval()
    rows = []
    tL_list, tB_list = [], []

    for _ in range(n_batches):
        instances = [
            generate_feasible_instance(
                m, device, tables, table_gammas, token_gammas, budget_mode,
                constraint_structure=constraint_structure,
                modulation_choices=modulation_choices,
            )
            for _ in range(batch_size)
        ]
        pb = collate_variable(instances, device=device)

        t0 = time.perf_counter()
        pL, nL, (_, muL) = model(pb, T=T_unroll)
        t1 = time.perf_counter()
        tL_list.append((t1 - t0) * 1000.0)

        JL = Jmean(model, pb, pL, nL).mean().item()
        kmL = kkt_metrics(model, pb, pL, nL, None, muL, p_active_tau)

        t2 = time.perf_counter()
        p0, n0, (_, mu0), _iters = mirror_prox_extragradient_primal_dual_stable(
            model, pb, T=1200, step_p=0.01, step_u=0.03, step_mu=0.05,
            mu_max=50.0, warm_feas_steps=80, init="feasible"
        )
        t3 = time.perf_counter()
        tB_list.append((t3 - t2) * 1000.0)

        J0 = Jmean(model, pb, p0, n0).mean().item()
        km0 = kkt_metrics(model, pb, p0, n0, None, mu0, p_active_tau)

        rows.append((JL, kmL, J0, km0))

    def avg(getter):
        return sum(getter(r) for r in rows) / len(rows)

    def avg_list(xs):
        return sum(xs) / max(len(xs), 1)

    learned = {"J": avg(lambda r: r[0]), **{k: avg(lambda r, kk=k: r[1][kk]) for k in rows[0][1].keys()}}
    mirror = {"J": avg(lambda r: r[2]), **{k: avg(lambda r, kk=k: r[3][kk]) for k in rows[0][3].keys()}}

    learned["runtime_ms"] = avg_list(tL_list)
    mirror["runtime_ms"] = avg_list(tB_list)

    return {"learned": learned, "mirrorprox": mirror}


@torch.no_grad()
def evaluate_constraint_structure_generalization(model, device, m_list, tables, table_gammas, token_gammas, budget_mode,
                                                 structures=("sparse", "group", "prefix", "dense"),
                                                 n_batches: int = 4, batch_size: int = 6, T_unroll: int = 64):
    results = {}
    for structure in structures:
        results[structure] = {}
        for m in m_list:
            out = eval_model_with_baseline(
                model, device, m, tables, table_gammas, token_gammas, budget_mode,
                n_batches=n_batches, batch_size=batch_size, T_unroll=T_unroll,
                constraint_structure=structure,
            )
            results[structure][m] = out
    return results


@torch.no_grad()
def evaluate_modulation_generalization(model, device, m_list, tables, table_gammas, token_gammas, budget_mode,
                                       modulation_groups: Optional[Dict[str, List[str]]] = None,
                                       n_batches: int = 4, batch_size: int = 6, T_unroll: int = 64,
                                       constraint_structure: str = "sparse"):
    if modulation_groups is None:
        modulation_groups = dict(TEST_MODULATION_GROUPS)

    results = {}
    for group_name, mods in modulation_groups.items():
        results[group_name] = {}
        for m in m_list:
            out = eval_model_with_baseline(
                model, device, m, tables, table_gammas, token_gammas, budget_mode,
                n_batches=n_batches, batch_size=batch_size, T_unroll=T_unroll,
                constraint_structure=constraint_structure,
                modulation_choices=mods,
            )
            results[group_name][m] = out
    return results
