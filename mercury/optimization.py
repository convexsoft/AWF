import torch

from .data import ProblemBatch
from .metrics import Jmean, inequality_violation, kkt_metrics
from .utils import get_analytical_nu, normalize_to_budget, proj_simplex_scaled, safe_softplus


@torch.no_grad()
def br_p_projected_primal_dual(
    model,
    pb: ProblemBatch,
    n_fixed: torch.Tensor,
    steps: int = 600,
    lr_p: float = 0.05,
    lr_mu: float = 0.25,
    tol_kkt: float = 3e-4,
    tol_ineq: float = 3e-5,
    mu_max: float = 200.0,
    p_active_tau: float = 5e-4,
):
    B, M = pb.beta.shape
    K = pb.A.shape[1]
    mask_m = pb.mask_m
    mask_k = pb.mask_k

    p = torch.zeros(B, M, device=pb.beta.device, dtype=torch.float32)
    p[mask_m] = 1.0
    p = p / p.sum(dim=-1, keepdim=True).clamp_min(1e-12) * pb.P.float()

    mu = torch.zeros(B, K, device=pb.beta.device, dtype=torch.float32)

    for t in range(steps):
        denom = (pb.sigma + n_fixed).clamp_min(1e-6)
        gamma = pb.beta * p / denom
        mm = model.mmse(gamma, pb.mod_id)
        At_mu = (pb.A.transpose(-1, -2) @ mu.unsqueeze(-1)).squeeze(-1) * mask_m.to(torch.float32)

        nabla_L = (mm * (pb.beta / denom) - At_mu) * mask_m.to(torch.float32)
        nu_star = get_analytical_nu(nabla_L, p, mask_m, p_active_tau)

        active = ((p > p_active_tau) & mask_m).to(torch.float32)
        inactive = (mask_m.to(torch.float32) - active).clamp_min(0.0)
        a_cnt = active.sum(dim=-1, keepdim=True).clamp_min(2.0)
        i_cnt = inactive.sum(dim=-1, keepdim=True).clamp_min(1.0)

        r_all = (nabla_L + nu_star) * mask_m.to(torch.float32)
        r_act = r_all * active
        r_inact = torch.relu(r_all) * inactive
        kkt_act = (r_act.abs().sum(dim=-1, keepdim=True) / a_cnt)
        kkt_inact = (r_inact.sum(dim=-1, keepdim=True) / i_cnt)
        kkt = (kkt_act + kkt_inact).mean().item()

        grad = (nabla_L + nu_star)
        p = proj_simplex_scaled(p + lr_p * grad, pb.P.float(), mask_m)

        Ap = (pb.A @ p.unsqueeze(-1)).squeeze(-1) * mask_k.to(torch.float32)
        mu = (mu + lr_mu * (Ap - pb.p_hat)).clamp(0.0, mu_max)

        ineq = inequality_violation(pb, p).mean().item()
        if (kkt < tol_kkt) and (ineq < tol_ineq) and (t > 40):
            break
    return p


@torch.no_grad()
def br_n_u_pg(
    model,
    pb: ProblemBatch,
    p_fixed: torch.Tensor,
    steps: int = 600,
    lr_u: float = 0.15,
    u_clip: float = 10.0,
    tol_level: float = 3e-4,
    n_restarts: int = 3,
    n_init=None,
):
    device = pb.beta.device
    dtype = torch.float32
    mask_m = pb.mask_m
    B, M = pb.beta.shape
    N_budget = pb.N.float()

    def objective(n_):
        return Jmean(model, pb, p_fixed, n_)

    def grad_terms(n_):
        denom = (pb.sigma + n_).clamp_min(1e-6)
        gamma = pb.beta * p_fixed / denom
        mm = model.mmse(gamma, pb.mod_id)
        g_i = (pb.beta * p_fixed) / denom.pow(2) * mm
        g_i = g_i * mask_m.to(dtype)
        grad_n = -g_i
        m_valid = mask_m.sum(dim=-1, keepdim=True).clamp_min(1).to(dtype)
        gbar = (g_i.sum(dim=-1, keepdim=True) / m_valid)
        level_res = ((g_i - gbar).abs().sum(dim=-1, keepdim=True) / m_valid).mean().item()
        return grad_n, level_res

    def proj_n(x):
        x = x.clamp_min(0.0) * mask_m.to(dtype)
        return normalize_to_budget(x, N_budget, mask_m)

    starts = []
    if n_init is not None:
        starts.append(proj_n(n_init.detach().to(dtype)))
    n0 = torch.zeros(B, M, device=device, dtype=dtype)
    n0[mask_m] = 1.0
    starts.append(proj_n(n0))

    denom0 = (pb.sigma + starts[0]).clamp_min(1e-6)
    gamma0 = pb.beta * p_fixed / denom0
    mm0 = model.mmse(gamma0, pb.mod_id)
    g0 = ((pb.beta * p_fixed) / denom0.pow(2) * mm0).clamp_min(0.0) * mask_m.to(dtype)
    starts.append(proj_n(g0 + 1e-8 * mask_m.to(dtype)))

    while len(starts) < max(1, n_restarts):
        z = torch.rand(B, M, device=device, dtype=dtype) * mask_m.to(dtype)
        starts.append(proj_n(z))

    best_n = starts[0]
    best_J = objective(best_n)

    for n in starts[:max(1, n_restarts)]:
        n = proj_n(n)
        curJ = objective(n)
        no_improve_rounds = 0

        for t in range(steps):
            grad_n, level_res = grad_terms(n)
            step = lr_u
            accepted = False
            for _ in range(12):
                cand = proj_n(n - step * grad_n)
                candJ = objective(cand)
                if torch.all(candJ <= curJ + 1e-10):
                    accepted = True
                    break
                step *= 0.5

            if accepted:
                if torch.all(candJ >= curJ - 1e-10):
                    no_improve_rounds += 1
                else:
                    no_improve_rounds = 0
                n, curJ = cand, candJ
            else:
                no_improve_rounds += 1

            if (level_res < tol_level and t > 40) or no_improve_rounds >= 20:
                break

        better = curJ < best_J
        best_n = torch.where(better[:, None], n, best_n)
        best_J = torch.minimum(best_J, curJ)

    return best_n


@torch.no_grad()
def mirror_prox_extragradient_primal_dual_stable(
    model,
    pb: ProblemBatch,
    T: int = 1200,
    step_p: float = 0.01,
    step_u: float = 0.05,
    step_mu: float = 0.05,
    u_clip: float = 10.0,
    mu_max: float = 50.0,
    warm_feas_steps: int = 80,
    init: str = "feasible",
    early_stop: bool = True,
    tol: float = 1e-2,
    check_every: int = 25,
    ineq_tol: float = 1e-3,
    p_active_tau: float = 5e-4,
):
    device = pb.beta.device
    beta, sigma = pb.beta.float(), pb.sigma.float()
    A, p_hat = pb.A.float(), pb.p_hat.float()
    mask_m, mask_k = pb.mask_m, pb.mask_k
    P, N = pb.P.float(), pb.N.float()
    mod_id = pb.mod_id

    B, M = beta.shape
    K = A.shape[1]
    m_valid = mask_m.sum(dim=-1, keepdim=True).clamp_min(1).float()

    if init == "feasible":
        p = torch.zeros(B, M, device=device)
        p[mask_m] = 1.0
        p = p / p.sum(dim=-1, keepdim=True).clamp_min(1e-12) * P

        n = torch.zeros(B, M, device=device)
        n[mask_m] = 1.0
        n = n / n.sum(dim=-1, keepdim=True).clamp_min(1e-12) * N
        u = torch.log(torch.expm1(n.clamp_min(1e-8))).clamp(-u_clip, u_clip) * mask_m.float()
    else:
        p = torch.rand(B, M, device=device) * mask_m.float()
        p = proj_simplex_scaled(p, P, mask_m)
        u = (0.1 * torch.randn(B, M, device=device)).clamp(-u_clip, u_clip) * mask_m.float()
        n = normalize_to_budget(safe_softplus(u) * mask_m.float(), N, mask_m)

    mu = torch.zeros(B, K, device=device)

    for _ in range(warm_feas_steps):
        Ap = (A @ p.unsqueeze(-1)).squeeze(-1) * mask_k.float()
        mu = (mu + 0.10 * (Ap - p_hat)).clamp(0.0, mu_max)
        At_mu = (A.transpose(-1, -2) @ mu.unsqueeze(-1)).squeeze(-1) * mask_m.float()
        p = proj_simplex_scaled((p - 0.01 * At_mu).clamp_min(0.0), P, mask_m)

    def F_eval(p_curr, u_curr, n_curr, mu_curr):
        denom = (sigma + n_curr).clamp_min(1e-6)
        gamma = beta * p_curr / denom
        mm = model.mmse(gamma, mod_id)
        At_mu = (A.transpose(-1, -2) @ mu_curr.unsqueeze(-1)).squeeze(-1) * mask_m.float()

        nabla_L_p = (mm * (beta / denom) - At_mu) * mask_m.float()
        nu_p_star = get_analytical_nu(nabla_L_p, p_curr, mask_m)
        grad_p = (nabla_L_p + nu_p_star) * mask_m.float()

        g_i = (beta * p_curr) / denom.pow(2) * mm
        g_i = g_i * mask_m.float()
        gbar = (g_i.sum(dim=-1, keepdim=True) / m_valid).detach()
        res_u = (g_i - gbar) * mask_m.float()

        Ap = (A @ p_curr.unsqueeze(-1)).squeeze(-1) * mask_k.float()
        grad_mu = (Ap - p_hat) * mask_k.float()
        return grad_p, res_u, grad_mu

    iters = 0
    for t in range(T):
        iters = t + 1
        gp, ru, gmu = F_eval(p, u, n, mu)

        p_bar = proj_simplex_scaled(p + step_p * gp, P, mask_m)
        u_bar = (u + step_u * ru).clamp(-u_clip, u_clip)
        n_bar = normalize_to_budget(safe_softplus(u_bar) * mask_m.float(), N, mask_m)
        mu_bar = (mu + step_mu * gmu).clamp(0.0, mu_max)

        gp_b, ru_b, gmu_b = F_eval(p_bar, u_bar, n_bar, mu_bar)

        p = proj_simplex_scaled(p + step_p * gp_b, P, mask_m)
        u = (u + step_u * ru_b).clamp(-u_clip, u_clip)
        n = normalize_to_budget(safe_softplus(u) * mask_m.float(), N, mask_m)
        mu = (mu + step_mu * gmu_b).clamp(0.0, mu_max)

        if early_stop and (iters % check_every == 0 or iters == 1):
            km = kkt_metrics(model, pb, p, n, None, mu, p_active_tau=p_active_tau)
            kkt_p = km.get("kkt_p_active", float("inf"))
            kkt_n = km.get("kkt_n", km.get("kkt_n_mean", float("inf")))
            ineq = km.get("ineq", float("inf"))
            if (kkt_p < tol) and (kkt_n < tol) and (ineq < ineq_tol):
                break

    nu_dummy = torch.zeros_like(mu[:, 0:1])
    return p, n, (nu_dummy, mu), iters
