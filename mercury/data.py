import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from .constants import MODULATION_ID_ORDER, MOD_ID_TO_NAME, MOD_NAME_TO_ID, TRAIN_MODULATION_CHOICES
from .qam import make_distribution_token


def sample_dirichlet_simplex(m: int, total: float, alpha: float, device: str):
    x = torch.distributions.Dirichlet(torch.full((m,), alpha, device=device)).sample()
    return x * total


def make_sparse_A(K: int, m: int, density_min: float = 0.02, density_max: float = 0.20, device: str = "cpu"):
    density = random.uniform(density_min, density_max)
    nnz = max(1, int(density * K * m))
    A_cpu = torch.zeros(K, m, device="cpu", dtype=torch.float32)
    rows = torch.randint(0, K, (nnz,), device="cpu")
    cols = torch.randint(0, m, (nnz,), device="cpu")
    vals = torch.rand(nnz, device="cpu")
    A_cpu.index_put_((rows, cols), vals, accumulate=True)
    A_cpu = A_cpu / A_cpu.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return A_cpu.to(device)


def make_group_A(K: int, m: int, device: str = "cpu"):
    K = max(1, min(K, m))
    perm = torch.randperm(m, device="cpu")
    splits = torch.chunk(perm, K)
    A_cpu = torch.zeros(K, m, device="cpu", dtype=torch.float32)
    for k, idx in enumerate(splits):
        if idx.numel() == 0:
            idx = perm[k:k + 1]
        A_cpu[k, idx] = 1.0 / max(int(idx.numel()), 1)
    return A_cpu.to(device)


def make_prefix_A(K: int, m: int, device: str = "cpu"):
    K = max(1, min(K, m))
    endpoints = torch.linspace(1, m, K, device="cpu")
    endpoints = torch.round(endpoints).long().clamp(1, m)
    A_cpu = torch.zeros(K, m, device="cpu", dtype=torch.float32)
    for k, end in enumerate(endpoints.tolist()):
        A_cpu[k, :end] = 1.0 / float(end)
    return A_cpu.to(device)


def make_dense_A(K: int, m: int, device: str = "cpu"):
    base = torch.rand(K, m, device="cpu", dtype=torch.float32)
    smooth = torch.rand(1, m, device="cpu", dtype=torch.float32)
    A_cpu = 0.65 * base + 0.35 * smooth
    mask = (torch.rand(K, m, device="cpu") > 0.05).to(torch.float32)
    A_cpu = A_cpu * mask
    A_cpu = A_cpu / A_cpu.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return A_cpu.to(device)


def build_constraint_matrix(K: int, m: int, structure: str = "sparse", device: str = "cpu"):
    structure = structure.lower()
    if structure == "sparse":
        return make_sparse_A(K, m, device=device)
    if structure == "group":
        return make_group_A(K, m, device=device)
    if structure in ("prefix", "cumulative"):
        return make_prefix_A(K, m, device=device)
    if structure in ("dense", "dense_correlated"):
        return make_dense_A(K, m, device=device)
    raise ValueError(f"Unknown constraint structure: {structure}")


def generate_feasible_instance(
    m: int,
    device: str,
    tables,
    table_gammas,
    token_gammas,
    budget_mode: str = "per_channel_fixed",
    constraint_structure: str = "sparse",
    P_range=(1.0, 10.0),
    N_range=(1.0, 10.0),
    Pbar_range=(0.05, 0.5),
    Nbar_range=(0.05, 0.5),
    K_ratio_min=0.05,
    K_ratio_max=0.30,
    p_hard=0.7,
    modulation_choices: Optional[List[str]] = None,
):
    assert budget_mode in ["total_fixed", "per_channel_fixed"]
    rhoK = random.uniform(K_ratio_min, K_ratio_max)
    K = max(1, int(round(rhoK * m)))

    if budget_mode == "total_fixed":
        P = random.uniform(*P_range)
        N = random.uniform(*N_range)
    else:
        P = random.uniform(*Pbar_range) * m
        N = random.uniform(*Nbar_range) * m

    beta = torch.exp(torch.randn(m, device=device) * 0.6)
    sigma = torch.exp(torch.randn(m, device=device) * 0.25) * 0.5
    A = build_constraint_matrix(K, m, structure=constraint_structure, device=device)

    p_feas = sample_dirichlet_simplex(m, total=P, alpha=1.0, device=device)
    Ap = A @ p_feas
    slack_lo, slack_hi = (0.02, 0.15) if random.random() < p_hard else (0.15, 0.40)
    slack = (slack_lo + (slack_hi - slack_lo) * torch.rand(K, device=device)) * (Ap.abs() + 1e-3)
    p_hat = Ap + slack

    modulation_choices = modulation_choices or TRAIN_MODULATION_CHOICES
    unknown_mods = [mod for mod in modulation_choices if mod not in MOD_NAME_TO_ID]
    if unknown_mods:
        raise ValueError(f"Unsupported modulation choices: {unknown_mods}. Supported: {list(MOD_NAME_TO_ID.keys())}")
    modulation = random.choice(modulation_choices)
    mod_id = MOD_NAME_TO_ID[modulation]
    d_token = make_distribution_token(modulation, token_gammas, table_gammas, tables).to(device)

    return {
        "m": m,
        "K": K,
        "beta": beta,
        "sigma": sigma,
        "A": A,
        "p_hat": p_hat,
        "P": torch.tensor([P], device=device),
        "N": torch.tensor([N], device=device),
        "d_token": d_token,
        "mod_id": torch.tensor(mod_id, device=device, dtype=torch.long),
        "constraint_structure": constraint_structure,
    }


@dataclass
class ProblemBatch:
    beta: torch.Tensor
    sigma: torch.Tensor
    mask_m: torch.Tensor
    A: torch.Tensor
    mask_k: torch.Tensor
    p_hat: torch.Tensor
    P: torch.Tensor
    N: torch.Tensor
    d_token: torch.Tensor
    mod_id: torch.Tensor


def modulation_histogram(mod_id: torch.Tensor) -> Dict[str, int]:
    counts = {name: 0 for name in MODULATION_ID_ORDER}
    for mid, cnt in zip(*torch.unique(mod_id.detach().cpu(), return_counts=True)):
        counts[MOD_ID_TO_NAME.get(int(mid.item()), str(int(mid.item())))] = int(cnt.item())
    return counts


def format_modulation_histogram(counts: Dict[str, int]) -> str:
    ordered = [f"{name}:{counts.get(name, 0)}" for name in MODULATION_ID_ORDER if name in counts]
    extras = [f"{name}:{counts[name]}" for name in counts if name not in MODULATION_ID_ORDER]
    return ",".join(ordered + extras)


def collate_variable(instances: List[Dict], device: str = "cpu") -> ProblemBatch:
    B = len(instances)
    Mmax = max(x["m"] for x in instances)
    Kmax = max(x["K"] for x in instances)

    beta = torch.zeros(B, Mmax, device=device)
    sigma = torch.zeros(B, Mmax, device=device)
    mask_m = torch.zeros(B, Mmax, dtype=torch.bool, device=device)
    A = torch.zeros(B, Kmax, Mmax, device=device)
    mask_k = torch.zeros(B, Kmax, dtype=torch.bool, device=device)
    p_hat = torch.zeros(B, Kmax, device=device)
    P = torch.zeros(B, 1, device=device)
    N = torch.zeros(B, 1, device=device)
    d_token = torch.zeros(B, 64, device=device)
    mod_id = torch.zeros(B, dtype=torch.long, device=device)

    for b, x in enumerate(instances):
        m, K = x["m"], x["K"]
        beta[b, :m] = x["beta"]
        sigma[b, :m] = x["sigma"]
        mask_m[b, :m] = True
        A[b, :K, :m] = x["A"]
        mask_k[b, :K] = True
        p_hat[b, :K] = x["p_hat"]
        P[b, 0] = x["P"]
        N[b, 0] = x["N"]
        d_token[b] = x["d_token"]
        mod_id[b] = x["mod_id"]

    return ProblemBatch(beta, sigma, mask_m, A, mask_k, p_hat, P, N, d_token, mod_id)


class MercuryDataset(Dataset):
    def __init__(self, n_samples, m_min, m_max, device, tables, table_gammas, token_gammas, budget_mode,
                 m_buckets: Optional[List[Tuple[int, int]]] = None,
                 bucket_probs: Optional[List[float]] = None,
                 modulation_choices: Optional[List[str]] = None):
        self.n = n_samples
        self.m_min = m_min
        self.m_max = m_max
        self.device = device
        self.tables = tables
        self.table_gammas = table_gammas
        self.token_gammas = token_gammas
        self.budget_mode = budget_mode
        self.modulation_choices = modulation_choices or TRAIN_MODULATION_CHOICES
        self.set_sampling_scheme(m_buckets=m_buckets, bucket_probs=bucket_probs)

    def set_sampling_scheme(self, m_buckets: Optional[List[Tuple[int, int]]] = None, bucket_probs: Optional[List[float]] = None):
        if not m_buckets:
            self.m_buckets = [(self.m_min, self.m_max)]
        else:
            self.m_buckets = [(max(self.m_min, int(lo)), min(self.m_max, int(hi))) for lo, hi in m_buckets]
        if bucket_probs is None:
            self.bucket_probs = [1.0 / len(self.m_buckets)] * len(self.m_buckets)
        else:
            probs = [float(max(0.0, p)) for p in bucket_probs]
            s = sum(probs)
            if s <= 0:
                raise ValueError("bucket_probs must sum to a positive number")
            self.bucket_probs = [p / s for p in probs]

    def sample_m(self):
        lo, hi = random.choices(self.m_buckets, weights=self.bucket_probs, k=1)[0]
        if hi < lo:
            lo, hi = hi, lo
        return random.randint(lo, hi)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        m = self.sample_m()
        return generate_feasible_instance(
            m=m,
            device=self.device,
            tables=self.tables,
            table_gammas=self.table_gammas,
            token_gammas=self.token_gammas,
            budget_mode=self.budget_mode,
            modulation_choices=self.modulation_choices,
        )


def expand_curriculum_phases(curriculum_phases, total_steps: int, default_m_min: int, default_m_max: int):
    if not curriculum_phases:
        return [{
            "until_frac": 1.0,
            "m_buckets": [(default_m_min, default_m_max)],
            "bucket_probs": [1.0],
            "modulation_choices": TRAIN_MODULATION_CHOICES,
        }]

    phases = []
    prev_until = 0.0
    for ph in curriculum_phases:
        until_frac = float(ph.get("until_frac", 1.0))
        until_frac = min(1.0, max(prev_until, until_frac))
        phases.append({
            "until_frac": until_frac,
            "m_buckets": ph.get("m_buckets", [(default_m_min, default_m_max)]),
            "bucket_probs": ph.get("bucket_probs"),
            "modulation_choices": ph.get("modulation_choices", TRAIN_MODULATION_CHOICES),
            "name": ph.get("name", f"phase_{len(phases)+1}"),
        })
        prev_until = until_frac
    if phases[-1]["until_frac"] < 1.0:
        phases[-1]["until_frac"] = 1.0
    return phases


def get_curriculum_phase(phases, step: int, total_steps: int):
    frac = step / max(1, total_steps)
    for ph in phases:
        if frac <= ph["until_frac"]:
            return ph
    return phases[-1]


def make_fixed_validation_buckets(device, tables, table_gammas, token_gammas, budget_mode,
                                  batch_size: int = 8, n_batches: int = 4,
                                  m_buckets: Optional[List[Tuple[int, int]]] = None,
                                  modulation_groups: Optional[Dict[str, List[str]]] = None):
    if m_buckets is None:
        m_buckets = [(32, 64), (96, 160), (224, 320), (384, 512)]
    if modulation_groups is None:
        modulation_groups = {
            "mod16": ["16QAM"],
            "mod64": ["64QAM"],
            "mixed": ["16QAM", "64QAM"],
        }

    buckets = {}
    for lo, hi in m_buckets:
        for group_name, mods in modulation_groups.items():
            key = f"m{lo}_{hi}_{group_name}"
            batches = []
            for _ in range(n_batches):
                instances = [
                    generate_feasible_instance(
                        m=random.randint(lo, hi), device=device, tables=tables,
                        table_gammas=table_gammas, token_gammas=token_gammas,
                        budget_mode=budget_mode, modulation_choices=mods,
                    )
                    for _ in range(batch_size)
                ]
                batches.append(collate_variable(instances, device=device))
            buckets[key] = batches
    return buckets
