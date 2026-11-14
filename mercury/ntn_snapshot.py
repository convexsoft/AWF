"""Frozen inputs for the NTN-inspired snapshot evaluation.

This module deliberately does not define a standardized coexistence model.
The desired-link operating point comes from 3GPP TR 38.811 Annex A and the
handheld receive assumptions come from 3GPP TR 38.821.  Interference severity
and the linear constraints remain explicitly algorithmic AWF settings.
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Sequence, Tuple

import torch

from .data import generate_feasible_instance


EXPERIMENT_LABEL = (
    "NTN-inspired snapshot evaluation — not a standardized coexistence scenario."
)

TR_38811_VERSION = "3GPP TR 38.811 V15.4.0 (2020-09), Release 15"
TR_38821_VERSION = "3GPP TR 38.821 V16.2.0 (2023-03), Release 16"

FINAL_BASE_SNAPSHOTS = 20
SMOKE_BASE_SNAPSHOTS = 3
BASE_SEED = 123
M = 64
INR_DB_LEVELS: Tuple[int, ...] = (-10, -5, 0, 5, 10)
MODULATION = "16QAM"

ALTITUDE_KM = 1500.0
EARTH_RADIUS_KM = 6371.0
SATELLITE_ECEF_KM = (7871.0, 0.0, 0.0)
B1_ECEF_KM = (6371.0, 0.0, 0.0)
B1_ELEVATION_DEG = 90.0
SLANT_RANGE_KM = 1500.0
CARRIER_FREQUENCY_GHZ = 2.0
TOTAL_BANDWIDTH_HZ = 5.0e6
BIN_BANDWIDTH_HZ = TOTAL_BANDWIDTH_HZ / M

EIRP_DENSITY_DBW_PER_MHZ = 36.0
P_PHYS_DBW = EIRP_DENSITY_DBW_PER_MHZ + 10.0 * math.log10(
    TOTAL_BANDWIDTH_HZ / 1.0e6
)
P_PHYS_W = 10.0 ** (P_PHYS_DBW / 10.0)

RX_ELEMENT_GAIN_DBI = 0.0
ANTENNA_TEMPERATURE_K = 290.0
AMBIENT_TEMPERATURE_K = 290.0
NOISE_FIGURE_DB = 7.0
NOISE_FACTOR = 10.0 ** (NOISE_FIGURE_DB / 10.0)
SYSTEM_NOISE_TEMPERATURE_K = ANTENNA_TEMPERATURE_K + (
    NOISE_FACTOR - 1.0
) * AMBIENT_TEMPERATURE_K
BOLTZMANN_DBW_PER_K_HZ = -228.6
BOLTZMANN_W_PER_K_HZ = 10.0 ** (BOLTZMANN_DBW_PER_K_HZ / 10.0)
SIGMA_BIN_PHYS_W = (
    BOLTZMANN_W_PER_K_HZ * SYSTEM_NOISE_TEMPERATURE_K * BIN_BANDWIDTH_HZ
)
SUM_SIGMA_PHYS_W = M * SIGMA_BIN_PHYS_W

FSPL_DB = (
    92.45
    + 20.0 * math.log10(CARRIER_FREQUENCY_GHZ)
    + 20.0 * math.log10(SLANT_RANGE_KM)
)
BETA_MEDIAN_PHYS = 10.0 ** (-FSPL_DB / 10.0)
SHADOW_STD_DB = 0.72

U_P_W = P_PHYS_W / (0.275 * M)
U_R_W = SIGMA_BIN_PHYS_W / 0.5
BETA_MEDIAN_PRIME = BETA_MEDIAN_PHYS * U_P_W / U_R_W
SIGMA_PRIME = SIGMA_BIN_PHYS_W / U_R_W
P_PRIME = P_PHYS_W / U_P_W
SUM_SIGMA_PRIME = M * SIGMA_PRIME


def rho_from_inr_db(inr_db: float) -> float:
    return 10.0 ** (float(inr_db) / 10.0)


def n_phys_from_inr_db(inr_db: float) -> float:
    return rho_from_inr_db(inr_db) * SUM_SIGMA_PHYS_W


def n_prime_from_inr_db(inr_db: float) -> float:
    return n_phys_from_inr_db(inr_db) / U_R_W


def beta_phys_from_shadow(shadow_db: float) -> float:
    return BETA_MEDIAN_PHYS * 10.0 ** (-float(shadow_db) / 10.0)


def beta_prime_from_shadow(shadow_db: float) -> float:
    return beta_phys_from_shadow(shadow_db) * U_P_W / U_R_W


def _inr_rows() -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for inr_db in INR_DB_LEVELS:
        n_phys = n_phys_from_inr_db(inr_db)
        rows.append({
            "INR_dB": float(inr_db),
            "rho_INR_aggregate": rho_from_inr_db(inr_db),
            "N_phys_W": n_phys,
            "N_phys_dBm": 10.0 * math.log10(n_phys) + 30.0,
            "N_prime": n_prime_from_inr_db(inr_db),
        })
    return rows


def frozen_provenance() -> Dict[str, Any]:
    """Return the complete pre-result provenance and interpretation contract."""
    return {
        "experiment_label": EXPERIMENT_LABEL,
        "frozen_before_smoke_results": True,
        "interpretation_rules": {
            "rho_INR": (
                "aggregate interference-budget-to-total-noise ratio "
                "N/sum_i(sigma_i), not a per-resource instantaneous INR"
            ),
            "paired_design": (
                "20 base snapshots are frozen before smoke review; every base snapshot "
                "is paired across all five INR levels and only N changes"
            ),
            "large_scale_shift": (
                "beta'_median=22.803252865 is a substantial large-scale shift relative "
                "to the synthetic beta=exp(N(0,0.6^2)) training scale (median 1); "
                "no result-dependent retraining or renormalization is permitted"
            ),
            "EIRP_budget": (
                "36 dBW/MHz is integrated over 5 MHz only to define an equivalent total "
                "AWF power budget; optimized bins are not claimed to satisfy a "
                "standardized 36 dBW/MHz spectral-density constraint"
            ),
            "constraints": (
                "A and p_hat are generic AWF resource-sharing constraints with no "
                "standardized NTN protection interpretation"
            ),
            "resource_bins": "64 equal-bandwidth AWF bins, not 3GPP PRBs",
            "channel_scope": (
                "large-scale operating-point snapshot, not a frequency-selective NTN "
                "channel simulator"
            ),
        },
        "sources": {
            "TR_38_811": {
                "version": TR_38811_VERSION,
                "main_snapshot": "Annex A, Table A-1",
                "earth_geometry": "Section 6.3",
                "free_space_path_loss": "Section 6.6.2, Eq. (6.6-2)",
                "LOS_probability": "Table 6.6.1-1",
                "LOS_shadow_fading": "Section 6.6.2 and Table 6.6.2-3",
                "D3_loss_scope": "Table 6.10.1-1",
            },
            "TR_38_821": {
                "version": TR_38821_VERSION,
                "handheld_RF": "Table 6.1.1.1-3, Handheld/S-band column",
                "downlink_polarization": (
                    "Section 6.1.1.1, text following Table 6.1.1.1-6"
                ),
                "link_budget_and_Boltzmann": "Section 6.1.3.1",
            },
            "repository_constraints": {
                "file": "mercury/data.py",
                "functions": [
                    "generate_feasible_instance",
                    "make_sparse_A",
                    "sample_dirichlet_simplex",
                ],
            },
        },
        "frozen_counts_and_seeds": {
            "final_base_snapshots": FINAL_BASE_SNAPSHOTS,
            "smoke_base_snapshots": SMOKE_BASE_SNAPSHOTS,
            "base_seed": BASE_SEED,
            "smoke_base_ids": list(range(SMOKE_BASE_SNAPSHOTS)),
            "final_base_ids": list(range(FINAL_BASE_SNAPSHOTS)),
            "INR_dB_levels": list(INR_DB_LEVELS),
            "paired_cases_smoke": SMOKE_BASE_SNAPSHOTS * len(INR_DB_LEVELS),
            "paired_cases_final": FINAL_BASE_SNAPSHOTS * len(INR_DB_LEVELS),
        },
        "physical_reference": {
            "scenario": "NGSO suburban B1 LOS large-scale calibration snapshot",
            "altitude_km": ALTITUDE_KM,
            "earth_radius_km": EARTH_RADIUS_KM,
            "satellite_ECEF_km": list(SATELLITE_ECEF_KM),
            "B1_ECEF_km": list(B1_ECEF_KM),
            "B1_elevation_deg": B1_ELEVATION_DEG,
            "slant_range_km": SLANT_RANGE_KM,
            "carrier_frequency_GHz": CARRIER_FREQUENCY_GHZ,
            "total_bandwidth_Hz": TOTAL_BANDWIDTH_HZ,
            "m_equal_bandwidth_AWF_bins": M,
            "bin_bandwidth_Hz": BIN_BANDWIDTH_HZ,
            "EIRP_density_dBW_per_MHz": EIRP_DENSITY_DBW_PER_MHZ,
            "P_phys_dBW": P_PHYS_DBW,
            "P_phys_W_EIRP": P_PHYS_W,
            "equal_reference_p_bin_phys_W_EIRP": P_PHYS_W / M,
            "Rx_element_gain_dBi": RX_ELEMENT_GAIN_DBI,
            "downlink_depolarization_loss_dB": 0.0,
            "antenna_temperature_K": ANTENNA_TEMPERATURE_K,
            "ambient_temperature_K": AMBIENT_TEMPERATURE_K,
            "noise_figure_dB": NOISE_FIGURE_DB,
            "noise_factor_linear": NOISE_FACTOR,
            "system_noise_temperature_K": SYSTEM_NOISE_TEMPERATURE_K,
            "Boltzmann_dBW_per_K_Hz": BOLTZMANN_DBW_PER_K_HZ,
            "Boltzmann_W_per_K_Hz": BOLTZMANN_W_PER_K_HZ,
            "sigma_bin_phys_W": SIGMA_BIN_PHYS_W,
            "sigma_bin_phys_dBm": 10.0 * math.log10(SIGMA_BIN_PHYS_W) + 30.0,
            "sum_sigma_phys_W": SUM_SIGMA_PHYS_W,
            "sum_sigma_phys_dBm": 10.0 * math.log10(SUM_SIGMA_PHYS_W) + 30.0,
            "FSPL_dB": FSPL_DB,
            "beta_median_phys": BETA_MEDIAN_PHYS,
            "shadow_fading_distribution_dB": "one common N(0, 0.72^2) draw per base snapshot",
            "shadow_fading_std_dB": SHADOW_STD_DB,
            "per_resource_shadow_draws": False,
            "LOS_conditioned": True,
            "fast_fading_modelled": False,
            "atmospheric_rain_cloud_losses": "excluded as negligible for the selected S-band scope",
            "ionospheric_scintillation": "excluded, not asserted to be zero",
        },
        "normalization": {
            "definition": {
                "p_phys": "u_p * p_prime",
                "n_phys": "u_r * n_prime",
                "sigma_phys": "u_r * sigma_prime",
                "beta_prime": "beta_phys * u_p / u_r",
            },
            "SINR_identity": (
                "beta_phys*p_phys/(sigma_phys+n_phys) == "
                "beta_prime*p_prime/(sigma_prime+n_prime)"
            ),
            "u_p_W_EIRP": U_P_W,
            "u_r_W": U_R_W,
            "beta_prime_median": BETA_MEDIAN_PRIME,
            "sigma_prime_per_bin": SIGMA_PRIME,
            "sum_sigma_prime": SUM_SIGMA_PRIME,
            "P_prime": P_PRIME,
            "equal_reference_p_prime_per_bin": P_PRIME / M,
        },
        "algorithmic_settings": {
            "modulation": MODULATION,
            "rho_INR_definition": "N / sum_i sigma_i",
            "INR_status": "predeclared algorithmic sensitivity settings, not 3GPP recommendations",
            "INR_rows": _inr_rows(),
            "constraint_structure": "sparse",
            "K_ratio_distribution": "Uniform[0.05, 0.30]",
            "K_integer_range_at_m64": [3, 19],
            "sparse_density_distribution": "Uniform[0.02, 0.20]",
            "A_nonzero_initial_distribution": "Uniform[0,1)",
            "A_row_normalization": "row sum equals 1 when nonempty",
            "p_feasible_distribution": "Dirichlet(ones), scaled to P'=17.6",
            "hard_slack_probability": 0.7,
            "hard_slack_relative_range": [0.02, 0.15],
            "regular_slack_relative_range": [0.15, 0.40],
            "p_hat_interpretation": "generic normalized AWF resource-sharing threshold",
        },
    }


def generate_base_snapshots(
    count: int,
    *,
    device: str,
    tables,
    table_gammas: torch.Tensor,
    token_gammas: torch.Tensor,
) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """Generate the frozen ordered base pool before expanding the INR sweep.

    The existing generic generator is used only for A and p_hat (plus the
    in-training 16QAM token).  Its synthetic beta, sigma and budgets are
    replaced by the sourced and frozen NTN values.
    """
    if count < 1 or count > FINAL_BASE_SNAPSHOTS:
        raise ValueError(
            f"count must be in [1, {FINAL_BASE_SNAPSHOTS}], received {count}"
        )

    # This sequential seed protocol makes the first three bases identical
    # whether the smoke subset or the complete frozen pool is constructed.
    random.seed(BASE_SEED)
    torch.manual_seed(BASE_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(BASE_SEED)

    generated: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    for base_id in range(count):
        generic = generate_feasible_instance(
            m=M,
            device=device,
            tables=tables,
            table_gammas=table_gammas,
            token_gammas=token_gammas,
            budget_mode="per_channel_fixed",
            constraint_structure="sparse",
            Pbar_range=(0.275, 0.275),
            Nbar_range=(0.5, 0.5),
            modulation_choices=[MODULATION],
        )
        shadow_db = random.gauss(0.0, SHADOW_STD_DB)
        beta_phys = beta_phys_from_shadow(shadow_db)
        beta_prime = beta_phys * U_P_W / U_R_W

        generic["beta"] = torch.full(
            (M,), beta_prime, device=device, dtype=torch.float32
        )
        generic["sigma"] = torch.full(
            (M,), SIGMA_PRIME, device=device, dtype=torch.float32
        )
        generic["P"] = torch.tensor([P_PRIME], device=device, dtype=torch.float32)
        generic["N"] = torch.tensor(
            [n_prime_from_inr_db(0.0)], device=device, dtype=torch.float32
        )

        metadata = {
            "base_snapshot_id": base_id,
            "pool_seed": BASE_SEED,
            "shadow_fading_common_dB": shadow_db,
            "beta_phys_all_bins": beta_phys,
            "beta_prime_all_bins_unrounded": beta_prime,
            "sigma_phys_all_bins_W": SIGMA_BIN_PHYS_W,
            "sigma_prime_all_bins": SIGMA_PRIME,
            "P_phys_W_EIRP": P_PHYS_W,
            "P_prime": P_PRIME,
            "K": int(generic["K"]),
        }
        generated.append((generic, metadata))
    return generated


def paired_instance(base_instance: Dict[str, Any], inr_db: int) -> Dict[str, Any]:
    """Clone one base snapshot and change only the aggregate N budget."""
    if inr_db not in INR_DB_LEVELS:
        raise ValueError(f"INR {inr_db} is not in the frozen grid {INR_DB_LEVELS}")
    paired: Dict[str, Any] = {}
    for key, value in base_instance.items():
        paired[key] = value.clone() if torch.is_tensor(value) else value
    paired["N"] = torch.tensor(
        [n_prime_from_inr_db(inr_db)],
        device=base_instance["N"].device,
        dtype=torch.float32,
    )
    return paired


def serialize_base_snapshot(
    base_instance: Dict[str, Any], metadata: Dict[str, Any]
) -> Dict[str, Any]:
    """Serialize all physical/normalized base inputs for provenance output."""
    beta_prime = base_instance["beta"].detach().cpu().double()
    sigma_prime = base_instance["sigma"].detach().cpu().double()
    A = base_instance["A"].detach().cpu().double()
    p_hat_prime = base_instance["p_hat"].detach().cpu().double()
    beta_phys = beta_prime * U_R_W / U_P_W
    sigma_phys = sigma_prime * U_R_W
    p_hat_phys = p_hat_prime * U_P_W
    return {
        **metadata,
        "beta_prime_float32": beta_prime.tolist(),
        "beta_phys_W_per_W_EIRP_from_stored_float32": beta_phys.tolist(),
        "sigma_prime_float32": sigma_prime.tolist(),
        "sigma_phys_W_from_stored_float32": sigma_phys.tolist(),
        "A_dimensionless": A.tolist(),
        "p_hat_prime": p_hat_prime.tolist(),
        "p_hat_equivalent_phys_W_EIRP": p_hat_phys.tolist(),
        "P_prime_float32": float(base_instance["P"].detach().cpu().item()),
        "P_equivalent_phys_W_EIRP_from_stored_float32": float(
            base_instance["P"].detach().cpu().item() * U_P_W
        ),
        "INR_budgets": _inr_rows(),
        "constraint_interpretation": (
            "generic AWF resource-sharing constraints; p_hat physical equivalents "
            "are unit conversions only, not standardized NTN protection limits"
        ),
    }
