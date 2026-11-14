"""Controlled single-component ablations of the final AWF architecture."""

from dataclasses import replace
from typing import List, Optional

import torch
import torch.nn as nn

from .constants import MODULATION_ID_ORDER
from .data import ProblemBatch
from .model import MercuryFoundationSolver


class IdentityConstraintMP(nn.Module):
    """Remove learned constraint-graph message passing and nothing else."""

    def forward(self, h, A, mask_m, mask_k):
        h_out = h * mask_m.unsqueeze(-1).to(h.dtype)
        constraint_features = torch.zeros(
            h.size(0), A.size(1), h.size(-1), device=h.device, dtype=h.dtype
        )
        constraint_features = constraint_features * mask_k.unsqueeze(-1).to(h.dtype)
        return h_out, constraint_features


class NoConstraintGNNSolver(MercuryFoundationSolver):
    """Final AWF with only BipartiteMP removed.

    ``super().__init__`` first constructs the complete model so this ablation
    consumes the same parameter-initialization RNG stream as the full model.
    Replacing ``mp`` afterwards removes only its trainable parameters.
    """

    def __init__(
        self,
        table_gammas: torch.Tensor,
        tables,
        modulation_id_order: Optional[List[str]] = None,
    ):
        super().__init__(
            table_gammas,
            tables,
            modulation_id_order or list(MODULATION_ID_ORDER),
        )
        self.mp = IdentityConstraintMP()


class NoDistributionTokenSolver(MercuryFoundationSolver):
    """Final AWF without encoder modulation/distribution conditioning.

    The 64-dimensional token input is identically zero, leaving encoder shape
    and allocated capacity unchanged.  ``mod_id`` remains available only to
    the physical MMSE lookup inside the optimization rollout; removing that
    lookup would change the mathematical problem rather than ablate encoder
    conditioning.
    """

    def forward(
        self,
        pb: ProblemBatch,
        T: int = 24,
        u_clip: float = 10.0,
        warm_feas_steps: Optional[int] = None,
    ):
        unconditioned = replace(pb, d_token=torch.zeros_like(pb.d_token))
        return super().forward(
            unconditioned,
            T=T,
            u_clip=u_clip,
            warm_feas_steps=warm_feas_steps,
        )


def build_ablation_solver(
    ablation: str,
    table_gammas: torch.Tensor,
    tables,
    modulation_id_order: Optional[List[str]] = None,
) -> MercuryFoundationSolver:
    names = modulation_id_order or list(MODULATION_ID_ORDER)
    if ablation == "no_constraint_gnn":
        return NoConstraintGNNSolver(table_gammas, tables, names)
    if ablation == "no_distribution_token":
        return NoDistributionTokenSolver(table_gammas, tables, names)
    raise ValueError(f"Unknown AWF ablation: {ablation}")

