from pathlib import Path
from typing import Any, Dict

import torch

from .constants import MODULATION_ID_ORDER
from .model import MercuryFoundationSolver


def extract_state_dict(checkpoint: Any) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must contain a state_dict-compatible mapping")
    return checkpoint


def checkpoint_tables(state_dict: Dict[str, torch.Tensor], device: torch.device):
    table_gammas = state_dict["table_gammas"].detach().float().to(device)
    tables = {
        name: {
            "mmse": state_dict["mmse_table"][index].detach().float().to(device),
            "I": state_dict["I_table"][index].detach().float().to(device),
        }
        for index, name in enumerate(MODULATION_ID_ORDER)
    }
    return table_gammas, tables


def load_awf_checkpoint(path: str | Path, device: str | torch.device = "cpu"):
    device = torch.device(device)
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    state_dict = extract_state_dict(checkpoint)
    table_gammas, tables = checkpoint_tables(state_dict, device)
    update_mode = checkpoint.get("n_update_mode", "projected")
    if update_mode != "projected":
        raise ValueError(
            "This public release supports the projected-n AWF checkpoint only; "
            f"checkpoint declares n_update_mode={update_mode!r}."
        )
    model = MercuryFoundationSolver(
        table_gammas=table_gammas,
        tables=tables,
        modulation_id_order=MODULATION_ID_ORDER,
        n_update_mode=update_mode,
    ).to(device=device, dtype=torch.float32)
    model.load_state_dict(state_dict, strict=True)
    model.heads.ap_max = float(checkpoint.get("alpha_p_max", model.heads.ap_max))
    model.heads.an_u_max = float(checkpoint.get("alpha_n_max", model.heads.an_u_max))
    model.heads.amu_max = float(checkpoint.get("alpha_mu_max", model.heads.amu_max))
    model.eval()
    return model, checkpoint, tables, table_gammas
