import torch

from .metrics import Jmean, inequality_violation, inequality_violation_sharp, n_kkt_regularizer, p_kkt_regularizer
from .optimization import br_n_u_pg, br_p_projected_primal_dual


def compute_loss(model, pb, p, n, mu,
                 lam_ineq, lam_kktp, lam_kktn,
                 p_active_tau, br_steps_p, br_steps_n,
                 lam_imit_p: float = 0.0, lam_imit_n: float = 0.0,
                 use_n_br_gap: bool = True):
    with torch.no_grad():
        p_br = br_p_projected_primal_dual(model, pb, n, steps=br_steps_p, lr_p=0.05, lr_mu=0.25, p_active_tau=p_active_tau)
        n_br = br_n_u_pg(model, pb, p, steps=br_steps_n, lr_u=0.15, n_init=n)

    J_pbr = Jmean(model, pb, p_br, n)
    J_nbr = Jmean(model, pb, p, n_br)
    J_cur = Jmean(model, pb, p, n)

    improve_p = torch.relu(J_pbr - J_cur)
    improve_n = torch.relu(J_cur - J_nbr)
    if use_n_br_gap:
        gap_loss = (improve_p + improve_n) / (J_cur.abs() + 1e-6)
    else:
        gap_loss = improve_p / (J_cur.abs() + 1e-6)
    gap_signed = (J_pbr - J_nbr) / (J_cur.abs() + 1e-6)

    mask = pb.mask_m.float()
    p_budget = pb.P.float().clamp_min(1e-6)
    n_budget = pb.N.float().clamp_min(1e-6)
    imit_p = (((p - p_br).abs() * mask).sum(dim=-1, keepdim=True) / p_budget).mean()
    imit_n = (((n - n_br).abs() * mask).sum(dim=-1, keepdim=True) / n_budget).mean()

    ineq = inequality_violation(pb, p)
    ineq_sharp = inequality_violation_sharp(pb, p)

    kktp = p_kkt_regularizer(model, pb, p, n, mu, p_active_tau)
    kktn = n_kkt_regularizer(model, pb, p, n)

    loss = (
        gap_loss.mean()
        + lam_ineq * ineq_sharp.mean()
        + lam_kktp * kktp
        + lam_kktn * kktn
        + lam_imit_p * imit_p
        + lam_imit_n * imit_n
    )

    return loss, {
        "loss": float(loss.item()),
        "gap": float(gap_signed.mean().item()),
        "gapLoss": float(gap_loss.mean().item()),
        "ineq": float(ineq.mean().item()),
        "ineqSharp": float(ineq_sharp.mean().item()),
        "Jmean": float(J_cur.mean().item()),
        "kktpReg": float(kktp.item()),
        "kktnReg": float(kktn.item()),
        "imitP": float(imit_p.item()),
        "imitN": float(imit_n.item()),
    }
