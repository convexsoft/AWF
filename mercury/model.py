from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants import MODULATION_ID_ORDER
from .data import ProblemBatch
from .utils import get_analytical_nu, interp_1d, normalize_to_budget, proj_simplex_scaled, safe_softplus, clamp_pos


class DistTokenMLP(nn.Module):
    def __init__(self, d_in: int = 64, d_out: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, 128), nn.ReLU(), nn.Linear(128, d_out), nn.ReLU())

    def forward(self, d):
        return self.net(d)


class CrossAttn(nn.Module):
    def __init__(self, d_model: int = 192, nhead: int = 8):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.ln_q = nn.LayerNorm(d_model)
        self.ln_kv = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model))
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, q, kv, kv_key_padding_mask=None):
        out, _ = self.attn(self.ln_q(q), self.ln_kv(kv), self.ln_kv(kv), key_padding_mask=kv_key_padding_mask)
        q = q + out
        q = q + self.ff(self.ln2(q))
        return q


class SelfAttnBlock(nn.Module):
    def __init__(self, d_model: int = 192, nhead: int = 8):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.ln = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model))
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        out, _ = self.attn(self.ln(x), self.ln(x), self.ln(x))
        x = x + out
        x = x + self.ff(self.ln2(x))
        return x


class PerceiverSetEncoder(nn.Module):
    def __init__(self, d_in, d_model: int = 192, n_latents: int = 64, n_layers: int = 4, nhead: int = 8):
        super().__init__()
        self.in_proj = nn.Linear(d_in, d_model)
        self.latents = nn.Parameter(torch.randn(1, n_latents, d_model) * 0.02)
        self.cross = nn.ModuleList([CrossAttn(d_model, nhead) for _ in range(n_layers)])
        self.selfs = nn.ModuleList([SelfAttnBlock(d_model, nhead) for _ in range(n_layers)])
        self.ch_to_lat = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.ln_ch = nn.LayerNorm(d_model)

    def forward(self, x, mask_m):
        B = x.size(0)
        h = self.in_proj(x)
        pad_mask = ~mask_m
        z = self.latents.expand(B, -1, -1)
        for ca, sa in zip(self.cross, self.selfs):
            z = ca(z, h, kv_key_padding_mask=pad_mask)
            z = sa(z)
        h2, _ = self.ch_to_lat(self.ln_ch(h), z, z)
        h = (h + h2) * mask_m.unsqueeze(-1).to(h.dtype)
        g = z.mean(dim=1)
        return h, g


class BipartiteMP(nn.Module):
    def __init__(self, d: int = 192):
        super().__init__()
        self.phi_c = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))
        self.phi_v = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))

    def forward(self, h, A, mask_m, mask_k):
        deg_k = A.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        c = (A @ h) / deg_k
        c = self.phi_c(c) * mask_k.unsqueeze(-1).to(c.dtype)
        deg_m = A.sum(dim=1).clamp_min(1e-6)
        msg = (A.transpose(-1, -2) @ c) / deg_m.unsqueeze(-1)
        h2 = self.phi_v(h + msg) * mask_m.unsqueeze(-1).to(h.dtype)
        return h2, c


class Heads(nn.Module):
    def __init__(self, d: int = 192, ap_max: float = 0.14, amu_max: float = 0.9, an_u_max: float = 0.6):
        super().__init__()
        self.p_logits = nn.Linear(d, 1)
        self.u0 = nn.Linear(d, 1)
        self.steps_raw = nn.Sequential(nn.Linear(d, 128), nn.ReLU(), nn.Linear(128, 3))
        self.ap_max, self.amu_max, self.an_u_max = ap_max, amu_max, an_u_max

    def forward(self, h, g, mask_m):
        p_log = self.p_logits(h).squeeze(-1)
        u = self.u0(h).squeeze(-1)
        p_log[~mask_m] = -1e9
        u[~mask_m] = 0.0

        s = torch.sigmoid(self.steps_raw(g))
        ap = self.ap_max * s[:, 0:1] + 1e-6
        amu = self.amu_max * s[:, 1:2] + 1e-6
        an_u = self.an_u_max * s[:, 2:3] + 1e-6
        return p_log, u, ap, amu, an_u


class MercuryFoundationSolver(nn.Module):
    def __init__(self, table_gammas, tables, modulation_id_order: Optional[List[str]] = None):
        super().__init__()
        self.dist_emb = DistTokenMLP(64, 64)
        self.enc = PerceiverSetEncoder(d_in=2 + 64, d_model=192, n_latents=64, n_layers=4, nhead=8)
        self.mp = BipartiteMP(d=192)
        self.heads = Heads(d=192)

        self.modulation_id_order = modulation_id_order or list(MODULATION_ID_ORDER)
        missing = [name for name in self.modulation_id_order if name not in tables]
        if missing:
            raise ValueError(f"Missing modulation tables for: {missing}")

        mmse_table = torch.stack([tables[name]["mmse"].float() for name in self.modulation_id_order], dim=0)
        I_table = torch.stack([tables[name]["I"].float() for name in self.modulation_id_order], dim=0)

        self.register_buffer("table_gammas", table_gammas.float())
        self.register_buffer("mmse_table", mmse_table)
        self.register_buffer("I_table", I_table)

    def mmse(self, gamma: torch.Tensor, mod_id: torch.Tensor) -> torch.Tensor:
        B, M = gamma.shape
        out = torch.empty_like(gamma, dtype=torch.float32)
        g = gamma.float()
        for mid in range(self.mmse_table.size(0)):
            mask = (mod_id == mid).view(B, 1).expand(B, M)
            if mask.any():
                out[mask] = interp_1d(g[mask], self.table_gammas, self.mmse_table[mid])
        return out

    def I(self, gamma: torch.Tensor, mod_id: torch.Tensor) -> torch.Tensor:
        B, M = gamma.shape
        out = torch.empty_like(gamma, dtype=torch.float32)
        g = gamma.float()
        for mid in range(self.I_table.size(0)):
            mask = (mod_id == mid).view(B, 1).expand(B, M)
            if mask.any():
                out[mask] = interp_1d(g[mask], self.table_gammas, self.I_table[mid])
        return out

    def forward(self, pb: ProblemBatch, T: int = 24, u_clip: float = 10.0):
        beta, sigma = pb.beta, pb.sigma
        A, p_hat = pb.A, pb.p_hat
        mask_m, mask_k = pb.mask_m, pb.mask_k
        P, N = pb.P, pb.N
        mod_id = pb.mod_id
        d_token = pb.d_token

        B, M = beta.shape
        m_valid = mask_m.sum(dim=-1, keepdim=True).clamp_min(1).to(beta.dtype)

        dist = self.dist_emb(d_token)
        dist_rep = dist[:, None, :].expand(B, M, dist.size(-1))
        x = torch.stack([torch.log(beta.clamp_min(1e-12)), torch.log(sigma.clamp_min(1e-12))], dim=-1)
        x = torch.cat([x, dist_rep], dim=-1)

        h, g = self.enc(x, mask_m)
        h, _ = self.mp(h, A, mask_m, mask_k)
        p_log, u, ap, amu, an_u = self.heads(h, g, mask_m)

        p = (P * F.softmax(p_log, dim=-1)) * mask_m.to(beta.dtype)
        u = u.clamp(-u_clip, u_clip) * mask_m.to(beta.dtype)
        n = safe_softplus(u) * mask_m.to(beta.dtype)
        n = normalize_to_budget(n, N.float(), mask_m)

        mu = torch.zeros(B, A.shape[1], device=beta.device, dtype=beta.dtype)

        warm_feas_steps = max(4, T // 4)
        for _ in range(warm_feas_steps):
            Ap = (A @ p.unsqueeze(-1)).squeeze(-1) * mask_k.to(beta.dtype)
            viol = (Ap - p_hat).clamp_min(0.0)
            mu = clamp_pos(mu + 2.0 * amu * viol)
            At_mu = (A.transpose(-1, -2) @ mu.unsqueeze(-1)).squeeze(-1) * mask_m.to(beta.dtype)
            p = proj_simplex_scaled((p - 0.50 * ap * At_mu).clamp_min(0.0), P.float(), mask_m)

        for _ in range(T):
            denom = (sigma + n).clamp_min(1e-6)
            gamma = beta * p / denom
            mm = self.mmse(gamma, mod_id)
            At_mu = (A.transpose(-1, -2) @ mu.unsqueeze(-1)).squeeze(-1) * mask_m.to(beta.dtype)

            nabla_L_p = (mm * (beta / denom) - At_mu) * mask_m.to(beta.dtype)
            nu_star = get_analytical_nu(nabla_L_p, p, mask_m)
            grad_p = (nabla_L_p + nu_star) * mask_m.to(beta.dtype)

            g_i = (beta * p) / denom.pow(2) * mm
            g_i = g_i * mask_m.to(beta.dtype)
            gbar = (g_i.sum(dim=-1, keepdim=True) / m_valid).detach()
            res_u = (g_i - gbar) * mask_m.to(beta.dtype)

            p_bar = proj_simplex_scaled(p + ap * grad_p, P.float(), mask_m)
            u_bar = (u + an_u * res_u).clamp(-u_clip, u_clip)
            n_bar = safe_softplus(u_bar) * mask_m.to(beta.dtype)
            n_bar = normalize_to_budget(n_bar, N.float(), mask_m)

            Ap_bar = (A @ p_bar.unsqueeze(-1)).squeeze(-1) * mask_k.to(beta.dtype)
            viol_bar = (Ap_bar - p_hat).clamp_min(0.0)
            mu_bar = clamp_pos(mu + 1.5 * amu * viol_bar)

            denom_b = (sigma + n_bar).clamp_min(1e-6)
            gamma_b = beta * p_bar / denom_b
            mm_b = self.mmse(gamma_b, mod_id)
            At_mu_b = (A.transpose(-1, -2) @ mu_bar.unsqueeze(-1)).squeeze(-1) * mask_m.to(beta.dtype)

            nabla_L_p_b = (mm_b * (beta / denom_b) - At_mu_b) * mask_m.to(beta.dtype)
            nu_star_b = get_analytical_nu(nabla_L_p_b, p_bar, mask_m)
            grad_p_b = (nabla_L_p_b + nu_star_b) * mask_m.to(beta.dtype)

            g_i_b = (beta * p_bar) / denom_b.pow(2) * mm_b
            g_i_b = g_i_b * mask_m.to(beta.dtype)
            gbar_b = (g_i_b.sum(dim=-1, keepdim=True) / m_valid).detach()
            res_u_b = (g_i_b - gbar_b) * mask_m.to(beta.dtype)

            p = proj_simplex_scaled(p + ap * grad_p_b, P.float(), mask_m)
            u = (u + an_u * res_u_b).clamp(-u_clip, u_clip)
            n = safe_softplus(u) * mask_m.to(beta.dtype)
            n = normalize_to_budget(n, N.float(), mask_m)

            Ap = (A @ p.unsqueeze(-1)).squeeze(-1) * mask_k.to(beta.dtype)
            viol = (Ap - p_hat).clamp_min(0.0)
            mu = clamp_pos(mu + 1.5 * amu * viol)

            At_mu_post = (A.transpose(-1, -2) @ mu.unsqueeze(-1)).squeeze(-1) * mask_m.to(beta.dtype)
            p = proj_simplex_scaled((p - 0.10 * ap * At_mu_post).clamp_min(0.0), P.float(), mask_m)

        return p, n, (torch.zeros_like(mu[:, 0:1]), mu)
