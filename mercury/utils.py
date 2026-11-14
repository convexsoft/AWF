import random
from typing import Optional

import torch
import torch.nn.functional as F


def seed_all(seed: int = 0) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def logspace_gamma_grid(gmin_db: float = -10.0, gmax_db: float = 30.0, n: int = 128, device: str = "cpu"):
    g_db = torch.linspace(gmin_db, gmax_db, n, device=device, dtype=torch.float32)
    return 10.0 ** (g_db / 10.0)


def interp_1d(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor):
    x = x.clamp(xp[0].item(), xp[-1].item())
    idx = torch.bucketize(x, xp).clamp(1, xp.numel() - 1)
    x0, x1 = xp[idx - 1], xp[idx]
    y0, y1 = fp[idx - 1], fp[idx]
    t = (x - x0) / (x1 - x0 + 1e-12)
    return y0 + t * (y1 - y0)


def clamp_pos(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp_min(x, 0.0)


def safe_softplus(u: torch.Tensor):
    return F.softplus(u.clamp(-20.0, 20.0))


def normalize_to_budget(x: torch.Tensor, budget: torch.Tensor, mask: torch.Tensor, eps: float = 1e-12):
    budget = budget.view(-1, 1).to(x.device, x.dtype)
    x = x * mask.to(x.dtype)
    s = x.sum(dim=-1, keepdim=True).clamp_min(eps)
    return x * (budget / s)


def proj_simplex_scaled(v: torch.Tensor, total: torch.Tensor, mask: torch.Tensor, eps: float = 1e-12):
    B, M = v.shape
    total = total.view(B, 1).to(v.device, v.dtype)
    v2 = v.clone()
    v2[~mask] = -1e9
    u, _ = torch.sort(v2, dim=-1, descending=True)
    cssv = torch.cumsum(u, dim=-1) - total
    ind = torch.arange(1, M + 1, device=v.device).view(1, -1).to(v.dtype)
    cond = u - cssv / ind > 0
    rho = cond.sum(dim=-1).clamp(min=1)
    rho_idx = (rho - 1).view(B, 1)
    theta = cssv.gather(1, rho_idx) / rho.view(B, 1).to(v.dtype)
    w = (v2 - theta).clamp_min(0.0)
    w[~mask] = 0.0
    s = w.sum(dim=-1, keepdim=True).clamp_min(eps)
    w = w * (total / s)
    return w


def hard_active_mask(p: torch.Tensor, mask_m: torch.Tensor, tau: float = 5e-4, min_active: int = 2):
    active = ((p > tau) & mask_m)
    if min_active <= 0:
        return active

    active = active.clone()
    B = p.size(0)
    for b in range(B):
        valid_idx = torch.where(mask_m[b])[0]
        if valid_idx.numel() == 0:
            continue
        cnt = int(active[b, valid_idx].sum().item())
        need = min(min_active, valid_idx.numel())
        if cnt < need:
            vals = p[b, valid_idx]
            top_idx_local = torch.topk(vals, k=need).indices
            top_idx = valid_idx[top_idx_local]
            active[b, top_idx] = True
    return active


def get_analytical_nu(nabla_L, p, mask_m, p_active_tau: float = 5e-4, min_active: int = 2):
    active = hard_active_mask(p, mask_m, tau=p_active_tau, min_active=min_active).to(torch.float32).detach()
    a_cnt = active.sum(dim=-1, keepdim=True).clamp_min(float(max(1, min_active)))
    nu_star = -(nabla_L * active).sum(dim=-1, keepdim=True) / a_cnt
    return nu_star
