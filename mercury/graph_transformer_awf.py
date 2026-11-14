"""Controlled Graph-Transformer replacement for AWF constraint message passing.

Only ``MercuryFoundationSolver.mp`` is replaced.  The inherited forward method
therefore keeps the original representation stack, heads, initialization,
feasibility warm-up, projections, and learned extragradient rollout unchanged.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from .constants import MODULATION_ID_ORDER
from .model import MercuryFoundationSolver


GRAPH_D_MODEL = 192
GRAPH_NUM_HEADS = 4
GRAPH_NUM_LAYERS = 1
GRAPH_D_FF = 192
GRAPH_DROPOUT = 0.0


class MaskedBipartiteGraphTransformer(nn.Module):
    """One post-norm Transformer block on resource and constraint tokens.

    Constraint tokens are formed without trainable parameters as ``A @ h``
    divided by the corresponding row sums.  On valid nodes the attention mask
    permits only self-loops and directed resource--constraint edges induced by
    ``A != 0``.  Padded query rows receive one numerical dummy edge to the first
    resource token and are zeroed immediately after each sublayer; this does not
    change the graph on valid nodes and prevents all-masked softmax rows.
    """

    def __init__(self) -> None:
        super().__init__()
        self.d_model = GRAPH_D_MODEL
        self.num_heads = GRAPH_NUM_HEADS
        self.num_layers = GRAPH_NUM_LAYERS
        self.d_ff = GRAPH_D_FF
        self.dropout = GRAPH_DROPOUT

        self.attn = nn.MultiheadAttention(
            embed_dim=GRAPH_D_MODEL,
            num_heads=GRAPH_NUM_HEADS,
            dropout=GRAPH_DROPOUT,
            bias=True,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(GRAPH_D_MODEL)
        self.ffn = nn.Sequential(
            nn.Linear(GRAPH_D_MODEL, GRAPH_D_FF),
            nn.GELU(),
            nn.Linear(GRAPH_D_FF, GRAPH_D_MODEL),
        )
        self.norm2 = nn.LayerNorm(GRAPH_D_MODEL)

    @staticmethod
    def constraint_tokens(
        h: torch.Tensor,
        A: torch.Tensor,
        mask_k: torch.Tensor,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        degree = A.sum(dim=-1, keepdim=True).clamp_min(eps)
        c = (A @ h) / degree
        return c * mask_k.unsqueeze(-1).to(h.dtype)

    @staticmethod
    def build_disallow_mask(
        A: torch.Tensor,
        mask_m: torch.Tensor,
        mask_k: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(disallow, valid)`` for the joint token sequence.

        ``disallow[b,q,k] == True`` follows the boolean-mask convention of
        ``nn.MultiheadAttention``.  For every valid query/key pair, allowed
        entries are exactly self-loops plus the two directions of ``A != 0``.
        """
        batch, constraints, resources = A.shape
        total = resources + constraints
        device = A.device
        valid = torch.cat((mask_m, mask_k), dim=1)
        allowed = torch.zeros(
            batch, total, total, dtype=torch.bool, device=device
        )

        valid_eye = (
            torch.eye(total, dtype=torch.bool, device=device)
            .unsqueeze(0)
            .expand(batch, -1, -1)
        )
        allowed |= valid_eye & valid.unsqueeze(1) & valid.unsqueeze(2)

        edges = (
            (A > 0.0)
            & mask_k.unsqueeze(-1)
            & mask_m.unsqueeze(1)
        )
        allowed[:, :resources, resources:] = edges.transpose(1, 2)
        allowed[:, resources:, :resources] = edges

        padded_queries = ~valid
        if padded_queries.any():
            # Every problem has at least one valid resource.  This edge exists
            # only for padded non-nodes, whose outputs are immediately zeroed.
            allowed[:, :, 0] |= padded_queries
        return ~allowed, valid

    def forward(
        self,
        h: torch.Tensor,
        A: torch.Tensor,
        mask_m: torch.Tensor,
        mask_k: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if h.size(-1) != GRAPH_D_MODEL:
            raise ValueError(
                f"Expected resource embedding width {GRAPH_D_MODEL}, got {h.size(-1)}"
            )
        resources = h.size(1)
        constraints = A.size(1)
        c = self.constraint_tokens(h, A, mask_k)
        x = torch.cat((h, c), dim=1)
        disallow, valid = self.build_disallow_mask(A, mask_m, mask_k)

        batch, total, _ = disallow.shape
        head_mask = (
            disallow[:, None, :, :]
            .expand(batch, self.num_heads, total, total)
            .reshape(batch * self.num_heads, total, total)
        )
        attn_out, _ = self.attn(
            x,
            x,
            x,
            attn_mask=head_mask,
            need_weights=False,
        )
        valid_f = valid.unsqueeze(-1).to(x.dtype)
        x = self.norm1(x + attn_out) * valid_f
        x = self.norm2(x + self.ffn(x)) * valid_f
        return x[:, :resources], x[:, resources:resources + constraints]


class GraphTransformerAWFSolver(MercuryFoundationSolver):
    """Final AWF architecture with only ``BipartiteMP`` replaced."""

    def __init__(
        self,
        table_gammas: torch.Tensor,
        tables,
        modulation_id_order: Optional[List[str]] = None,
    ) -> None:
        # Constructing the full parent first preserves the initialization RNG
        # stream of every common module; only ``mp`` is then replaced.
        super().__init__(
            table_gammas,
            tables,
            modulation_id_order or list(MODULATION_ID_ORDER),
        )
        self.mp = MaskedBipartiteGraphTransformer()


def trainable_parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)

