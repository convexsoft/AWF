"""Low-complexity direct neural baseline for the AWF experiments.

The baseline intentionally reuses the proposed solver's input/token encoder and
permutation-compatible graph representation, but has no learned optimization
rollout.  It predicts a complete primal-dual state in one pass and enforces the
two simplex budgets deterministically.
"""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants import MODULATION_ID_ORDER
from .data import ProblemBatch
from .model import MercuryFoundationSolver
from .utils import normalize_to_budget, safe_softplus


class DirectBipartiteMP(nn.Module):
    """One permutation-compatible channel/constraint message-passing layer.

    ``p_hat`` and the two budgets are available to the proposed AWF rollout.
    Tiny scalar embeddings expose the same information to the direct decoder;
    they do not introduce any iterative optimization computation.
    """

    def __init__(self, d: int = 192):
        super().__init__()
        self.budget_embed = nn.Linear(2, d)
        self.threshold_embed = nn.Linear(1, d)
        self.phi_c = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))
        self.phi_v = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))

    def forward(
        self,
        h: torch.Tensor,
        A: torch.Tensor,
        p_hat: torch.Tensor,
        P: torch.Tensor,
        N: torch.Tensor,
        mask_m: torch.Tensor,
        mask_k: torch.Tensor,
    ):
        dtype = h.dtype
        mask_m_f = mask_m.unsqueeze(-1).to(dtype)
        mask_k_f = mask_k.unsqueeze(-1).to(dtype)
        m_valid = mask_m.sum(dim=-1, keepdim=True).clamp_min(1).to(dtype)

        p_per_channel = (P.to(dtype) / m_valid).clamp_min(1e-8)
        n_per_channel = (N.to(dtype) / m_valid).clamp_min(1e-8)
        budget_features = torch.cat(
            [torch.log(p_per_channel), torch.log(n_per_channel)], dim=-1
        )
        h_ctx = (h + self.budget_embed(budget_features).unsqueeze(1)) * mask_m_f

        deg_k = A.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        c_aggregate = (A @ h_ctx) / deg_k
        normalized_threshold = torch.log(
            (p_hat.to(dtype) / p_per_channel).clamp(1e-6, 1e6)
        ).unsqueeze(-1)
        c = self.phi_c(c_aggregate + self.threshold_embed(normalized_threshold))
        c = c * mask_k_f

        deg_m = A.sum(dim=1).clamp_min(1e-6)
        message = (A.transpose(-1, -2) @ c) / deg_m.unsqueeze(-1)
        h_out = self.phi_v(h_ctx + message) * mask_m_f
        return h_out, c


class DirectHeads(nn.Module):
    """One-shot primal/dual prediction heads with deterministic constraints."""

    def __init__(self, d: int = 192):
        super().__init__()
        self.p_logits = nn.Linear(d, 1)
        self.n_raw = nn.Linear(d, 1)
        self.mu_raw = nn.Linear(d, 1)

        # The proposed solver starts mu at zero.  A small positive direct
        # initialization avoids a large arbitrary dual offset before training.
        nn.init.constant_(self.mu_raw.bias, -5.0)

    def forward(
        self,
        h: torch.Tensor,
        c: torch.Tensor,
        P: torch.Tensor,
        N: torch.Tensor,
        mask_m: torch.Tensor,
        mask_k: torch.Tensor,
    ):
        dtype = h.dtype
        mask_m_f = mask_m.to(dtype)
        mask_k_f = mask_k.to(dtype)

        p_logits = self.p_logits(h).squeeze(-1).masked_fill(~mask_m, -1e9)
        p = P.to(dtype) * F.softmax(p_logits, dim=-1)
        p = p * mask_m_f

        n_raw = self.n_raw(h).squeeze(-1)
        n_positive = safe_softplus(n_raw) * mask_m_f
        n = normalize_to_budget(n_positive, N.to(dtype), mask_m)

        mu = safe_softplus(self.mu_raw(c).squeeze(-1)) * mask_k_f
        return p, n, mu


class DirectGNNSolver(MercuryFoundationSolver):
    """Direct-GNN baseline with no primal-dual/extragradient iterations."""

    def __init__(
        self,
        table_gammas: torch.Tensor,
        tables,
        modulation_id_order: Optional[List[str]] = None,
    ):
        # Reuse the exact distribution-token and Perceiver encoder
        # implementations, as well as the official MMSE/I table interface used
        # by the shared training loss and evaluation metrics.
        super().__init__(
            table_gammas=table_gammas,
            tables=tables,
            modulation_id_order=modulation_id_order or list(MODULATION_ID_ORDER),
        )
        self.mp = DirectBipartiteMP(d=192)
        self.heads = DirectHeads(d=192)

    def forward(self, pb: ProblemBatch):
        beta, sigma = pb.beta, pb.sigma
        mask_m = pb.mask_m

        dist = self.dist_emb(pb.d_token)
        dist_rep = dist[:, None, :].expand(beta.size(0), beta.size(1), dist.size(-1))
        x = torch.stack(
            [torch.log(beta.clamp_min(1e-12)), torch.log(sigma.clamp_min(1e-12))],
            dim=-1,
        )
        x = torch.cat([x, dist_rep], dim=-1)

        h, _ = self.enc(x, mask_m)
        h, c = self.mp(
            h,
            pb.A,
            pb.p_hat,
            pb.P,
            pb.N,
            pb.mask_m,
            pb.mask_k,
        )
        p, n, mu = self.heads(h, c, pb.P, pb.N, pb.mask_m, pb.mask_k)
        return p, n, (torch.zeros_like(mu[:, 0:1]), mu)

