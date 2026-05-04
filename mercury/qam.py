import math
from typing import Dict, Optional

import torch

from .utils import interp_1d


def qam_constellation(M: int, device: str):
    s = int(math.isqrt(M))
    assert s * s == M
    levels = torch.arange(-(s - 1), s, 2, device=device, dtype=torch.float32)
    re, im = torch.meshgrid(levels, levels, indexing="ij")
    x = (re.flatten() + 1j * im.flatten()).to(torch.complex64)
    x = x / torch.sqrt((x.abs() ** 2).mean())
    return x


@torch.no_grad()
def simulate_mmse_I_for_gamma(constel: torch.Tensor, gamma: float, n_samples: int, device: str):
    M = constel.numel()
    idx = torch.randint(0, M, (n_samples,), device=device)
    x = constel[idx]
    n = (torch.randn(n_samples, device=device) + 1j * torch.randn(n_samples, device=device)) / math.sqrt(2.0)
    y = math.sqrt(gamma) * x + n

    diff = y[:, None] - math.sqrt(gamma) * constel[None, :]
    ll = -(diff.abs() ** 2)
    lse = torch.logsumexp(ll, dim=-1, keepdim=True)
    post = torch.exp(ll - lse)

    xhat = (post * constel[None, :]).sum(dim=-1)
    mmse = ((x - xhat).abs() ** 2).mean().item()

    ll_true = ll.gather(1, idx.view(-1, 1)).squeeze(1)
    log_py = torch.logsumexp(ll, dim=-1) - math.log(M)
    I_bits = ((ll_true - log_py).mean() / math.log(2.0)).item()
    return mmse, I_bits


@torch.no_grad()
def build_tables(
    device: str,
    gammas: torch.Tensor,
    n_samples_per_gamma: int = 15000,
    modulation_orders: Optional[Dict[str, int]] = None,
):
    from .constants import SUPPORTED_MODULATIONS

    modulation_orders = modulation_orders or SUPPORTED_MODULATIONS
    tables = {}
    for name, M in modulation_orders.items():
        constel = qam_constellation(M, device=device)
        mmse_list, I_list = [], []
        for g in gammas.tolist():
            mm, Ib = simulate_mmse_I_for_gamma(constel, g, n_samples_per_gamma, device=device)
            mmse_list.append(mm)
            I_list.append(Ib)
        tables[name] = {
            "mmse": torch.tensor(mmse_list, device=device, dtype=torch.float32),
            "I": torch.tensor(I_list, device=device, dtype=torch.float32),
        }
    return tables


def make_distribution_token(modulation: str, token_gammas, table_gammas, tables):
    mmse_vals = interp_1d(token_gammas, table_gammas, tables[modulation]["mmse"])
    I_vals = interp_1d(token_gammas, table_gammas, tables[modulation]["I"])
    return torch.cat([mmse_vals, I_vals], dim=0).to(torch.float32)
