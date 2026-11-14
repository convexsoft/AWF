import torch

from .data import ProblemBatch
from .utils import get_analytical_nu, hard_active_mask, proj_simplex_scaled


R_STAT_EVAL_STEP = 1e-2


@torch.no_grad()
def Jmean(
    model,
    pb: ProblemBatch,
    p: torch.Tensor,
    n: torch.Tensor,
    utility_weights: torch.Tensor = None,
) -> torch.Tensor:
    denom = (pb.sigma + n).clamp_min(1e-6)
    gamma = pb.beta * p / denom
    I = model.I(gamma, pb.mod_id)
    if utility_weights is not None:
        I = I * utility_weights.to(device=I.device, dtype=I.dtype)
    I = I * pb.mask_m.to(torch.float32)
    m_valid = pb.mask_m.sum(dim=-1).clamp_min(1).to(torch.float32)
    return I.sum(dim=-1) / m_valid


@torch.no_grad()
def inequality_violation(pb: ProblemBatch, p: torch.Tensor) -> torch.Tensor:
    Ap = (pb.A @ p.unsqueeze(-1)).squeeze(-1) * pb.mask_k.to(torch.float32)
    viol = (Ap - pb.p_hat).clamp_min(0.0) * pb.mask_k.to(torch.float32)
    k_valid = pb.mask_k.sum(dim=-1).clamp_min(1).to(torch.float32)
    return viol.sum(dim=-1) / k_valid


def inequality_violation_sharp(pb: ProblemBatch, p: torch.Tensor) -> torch.Tensor:
    Ap = (pb.A @ p.unsqueeze(-1)).squeeze(-1) * pb.mask_k.to(torch.float32)
    viol = (Ap - pb.p_hat).clamp_min(0.0) * pb.mask_k.to(torch.float32)
    k_valid = pb.mask_k.sum(dim=-1).clamp_min(1).to(torch.float32)
    mean_viol = viol.sum(dim=-1) / k_valid
    max_viol = viol.max(dim=-1).values
    sq_viol = (viol.square().sum(dim=-1) / k_valid)
    return mean_viol + 0.50 * max_viol + 0.25 * sq_viol


@torch.no_grad()
def projected_stationarity_residual(
    model,
    pb: ProblemBatch,
    p: torch.Tensor,
    n: torch.Tensor,
    mu: torch.Tensor,
    stat_step: float = R_STAT_EVAL_STEP,
    utility_weights: torch.Tensor = None,
) -> torch.Tensor:
    """Dimension-normalized projected residual of the full (p, n, mu) state."""
    if stat_step <= 0.0:
        raise ValueError("stat_step must be positive")

    mask_m, mask_k = pb.mask_m, pb.mask_k
    mask_m_f, mask_k_f = mask_m.to(torch.float32), mask_k.to(torch.float32)
    beta, sigma = pb.beta.float(), pb.sigma.float()
    p_f, n_f, mu_f = p.float(), n.float(), mu.float()
    A, p_hat = pb.A.float(), pb.p_hat.float()

    denom = (sigma + n_f).clamp_min(1e-6)
    gamma = beta * p_f / denom
    mm = model.mmse(gamma, pb.mod_id)
    if utility_weights is not None:
        weights = utility_weights.to(device=mm.device, dtype=mm.dtype)
        if weights.shape != mm.shape:
            raise ValueError(f"utility_weights must have shape {tuple(mm.shape)}, got {tuple(weights.shape)}")
        mm = mm * weights

    At_mu = (A.transpose(-1, -2) @ mu_f.unsqueeze(-1)).squeeze(-1)
    d_p = (mm * (beta / denom) - At_mu) * mask_m_f
    g_n = ((beta * p_f) / denom.pow(2) * mm) * mask_m_f
    h_mu = ((A @ p_f.unsqueeze(-1)).squeeze(-1) - p_hat) * mask_k_f

    p_proj = proj_simplex_scaled(p_f + stat_step * d_p, pb.P.float(), mask_m)
    n_proj = proj_simplex_scaled(n_f + stat_step * g_n, pb.N.float(), mask_m)
    mu_proj = (mu_f + stat_step * h_mu).clamp_min(0.0) * mask_k_f

    r_p = ((p_f - p_proj) / stat_step) * mask_m_f
    r_n = ((n_f - n_proj) / stat_step) * mask_m_f
    r_mu = ((mu_f * mask_k_f - mu_proj) / stat_step) * mask_k_f

    residual_sq = r_p.square().sum(dim=-1) + r_n.square().sum(dim=-1) + r_mu.square().sum(dim=-1)
    state_dim = (2 * mask_m.sum(dim=-1) + mask_k.sum(dim=-1)).clamp_min(1).to(torch.float32)
    return torch.sqrt(residual_sq / state_dim)


@torch.no_grad()
def kkt_metrics(
    model,
    pb: ProblemBatch,
    p,
    n,
    _,
    mu,
    p_active_tau: float = 5e-4,
    utility_weights: torch.Tensor = None,
):
    beta, sigma = pb.beta, pb.sigma
    A, p_hat = pb.A, pb.p_hat
    mask_m, mask_k = pb.mask_m, pb.mask_k
    m_valid = mask_m.sum(dim=-1, keepdim=True).clamp_min(1).to(torch.float32)
    k_valid = mask_k.sum(dim=-1, keepdim=True).clamp_min(1).to(torch.float32)

    denom = (sigma + n).clamp_min(1e-6)
    gamma = beta * p / denom
    mm = model.mmse(gamma, pb.mod_id)
    if utility_weights is not None:
        weights = utility_weights.to(device=mm.device, dtype=mm.dtype)
        if weights.shape != mm.shape:
            raise ValueError(f"utility_weights must have shape {tuple(mm.shape)}, got {tuple(weights.shape)}")
        mm = mm * weights
    At_mu = (A.transpose(-1, -2) @ mu.unsqueeze(-1)).squeeze(-1)

    nabla_L = (mm * (beta / denom) - At_mu) * mask_m.to(torch.float32)
    nu_star = get_analytical_nu(nabla_L, p, mask_m, p_active_tau)

    r_all = (nabla_L + nu_star) * mask_m.to(torch.float32)
    kkt_p_all = (r_all.abs().sum(dim=-1, keepdim=True) / m_valid).mean().item()

    active = hard_active_mask(p, mask_m, tau=p_active_tau, min_active=2).to(torch.float32)
    a_cnt = active.sum(dim=-1, keepdim=True).clamp_min(2.0)
    r_act = (nabla_L + nu_star) * active
    kkt_p_active = (r_act.abs().sum(dim=-1, keepdim=True) / a_cnt).mean().item()

    inactive = (mask_m.to(torch.float32) - active).clamp_min(0.0)
    i_cnt = inactive.sum(dim=-1, keepdim=True).clamp_min(1.0)
    r_inact = torch.relu(nabla_L + nu_star) * inactive
    kkt_p_inactive = (r_inact.sum(dim=-1, keepdim=True) / i_cnt).mean().item()

    g_i = (beta * p) / denom.pow(2) * mm
    g_i = g_i * mask_m.to(torch.float32)

    n_active_tau = 1e-4
    n_active = ((n > n_active_tau) & mask_m).to(torch.float32)
    n_inactive = (mask_m.to(torch.float32) - n_active).clamp_min(0.0)

    na_cnt = n_active.sum(dim=-1, keepdim=True).clamp_min(1.0)
    gbar = (g_i * n_active).sum(dim=-1, keepdim=True) / na_cnt

    r_n_act = (g_i - gbar).abs() * n_active
    kkt_n_active = (r_n_act.sum(dim=-1, keepdim=True) / na_cnt).mean().item()

    r_n_ineq = torch.relu(g_i - gbar) * n_inactive
    ni_cnt = n_inactive.sum(dim=-1, keepdim=True).clamp_min(1.0)
    kkt_n_ineq = (r_n_ineq.sum(dim=-1, keepdim=True) / ni_cnt).mean().item()

    kkt_n = kkt_n_active + kkt_n_ineq

    Ap = (A @ p.unsqueeze(-1)).squeeze(-1) * mask_k.to(torch.float32)
    viol = (Ap - p_hat).clamp_min(0.0)
    ineq = (viol.sum(dim=-1, keepdim=True) / k_valid).mean().item()

    sumP = (p.sum(dim=-1, keepdim=True) - pb.P.float()).abs().mean().item()
    sumN = (n.sum(dim=-1, keepdim=True) - pb.N.float()).abs().mean().item()

    return {
        "kkt_p_all": kkt_p_all,
        "kkt_p_active": kkt_p_active,
        "kkt_p_inactive": kkt_p_inactive,
        "kkt_n": kkt_n,
        "kkt_n_active": kkt_n_active,
        "kkt_n_ineq": kkt_n_ineq,
        "ineq": ineq,
        "sumP_abs": sumP,
        "sumN_abs": sumN,
        "mu_max": float(mu.max().item()) if mu.numel() else 0.0,
        "nu_max": float(nu_star.abs().max().item()) if nu_star.numel() else 0.0,
    }


def p_kkt_regularizer(model, pb: ProblemBatch, p, n, mu, p_active_tau: float = 5e-4):
    beta, sigma = pb.beta, pb.sigma
    A = pb.A
    mask_m = pb.mask_m

    denom = (sigma + n).clamp_min(1e-6)
    gamma = beta * p / denom
    mm = model.mmse(gamma, pb.mod_id)
    At_mu = (A.transpose(-1, -2) @ mu.unsqueeze(-1)).squeeze(-1)

    nabla_L = mm * (beta / denom) - At_mu
    nu_star = get_analytical_nu(nabla_L, p, mask_m, p_active_tau)

    active = hard_active_mask(p, mask_m, tau=p_active_tau, min_active=2).to(nabla_L.dtype).detach()
    a_cnt = active.sum(dim=-1, keepdim=True).clamp_min(2.0)
    res = (nabla_L + nu_star) * active
    return (res.abs().sum(dim=-1, keepdim=True) / a_cnt).mean()


@torch.no_grad()
def n_waterlevel_terms(model, pb: ProblemBatch, p: torch.Tensor, n: torch.Tensor, n_active_tau: float = 1e-4):
    beta, sigma = pb.beta, pb.sigma
    mask_m = pb.mask_m
    denom = (sigma + n).clamp_min(1e-6)
    gamma = beta * p / denom
    mm = model.mmse(gamma, pb.mod_id)
    g_i = (beta * p) / denom.pow(2) * mm
    g_i = g_i * mask_m.to(torch.float32)

    n_active = ((n > n_active_tau) & mask_m).to(torch.float32)
    n_inactive = (mask_m.to(torch.float32) - n_active).clamp_min(0.0)
    na_cnt = n_active.sum(dim=-1, keepdim=True).clamp_min(1.0)
    ni_cnt = n_inactive.sum(dim=-1, keepdim=True).clamp_min(1.0)
    gbar = (g_i * n_active).sum(dim=-1, keepdim=True) / na_cnt
    r_n_act = (g_i - gbar) * n_active
    r_n_ineq = torch.relu(g_i - gbar) * n_inactive
    return g_i, gbar, n_active, n_inactive, na_cnt, ni_cnt, r_n_act, r_n_ineq


def n_kkt_regularizer(model, pb, p, n, n_active_tau: float = 1e-4):
    _, _, _, _, na_cnt, ni_cnt, r_n_act, r_n_ineq = n_waterlevel_terms(model, pb, p, n, n_active_tau)
    act_term = (r_n_act.abs().sum(dim=-1, keepdim=True) / na_cnt)
    ineq_term = (r_n_ineq.sum(dim=-1, keepdim=True) / ni_cnt)
    return (act_term + ineq_term).mean()
