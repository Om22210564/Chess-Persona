from pathlib import Path
from typing import Optional, Union

import torch
from huggingface_hub import hf_hub_download
from maia3.models import MAIA3Model

from model_config import get_maia3_5m_config


DEFAULT_REPO_ID = "UofTCSSLab/Maia3-5M"
DEFAULT_CHECKPOINT_FILENAME = "maia3-5m.pt"


def download_maia3_checkpoint(
    repo_id: str = DEFAULT_REPO_ID,
    filename: str = DEFAULT_CHECKPOINT_FILENAME,
) -> str:
    """Download or reuse the cached Maia3 checkpoint from Hugging Face."""
    return hf_hub_download(repo_id=repo_id, filename=filename)


def _load_state_dict(path: Union[str, Path], map_location="cpu"):
    checkpoint = torch.load(path, map_location=map_location)
    return {k.replace("smolgen", "gab"): v for k, v in checkpoint.items()}


def load_maia3_model(
    device: Union[str, torch.device] = "cpu",
    finetuned_path: Optional[Union[str, Path]] = None,
    repo_id: str = DEFAULT_REPO_ID,
    checkpoint_filename: str = DEFAULT_CHECKPOINT_FILENAME,
) -> MAIA3Model:
    """
    Load Maia3-5M and optionally overlay a fine-tuned state dict.

    Parameters
    ----------
    device:
        Target device, e.g. "cpu" or "cuda".
    finetuned_path:
        Optional local checkpoint produced by training.
    repo_id/checkpoint_filename:
        Hugging Face checkpoint source for the base Maia3 model.
    """
    cfg = get_maia3_5m_config()
    model = MAIA3Model(cfg)

    base_path = download_maia3_checkpoint(repo_id, checkpoint_filename)
    base_state = _load_state_dict(base_path, map_location="cpu")
    model.load_state_dict(base_state, strict=False)

    if finetuned_path is not None:
        finetuned_path = Path(finetuned_path)
        if not finetuned_path.exists():
            raise FileNotFoundError(f"Fine-tuned checkpoint not found: {finetuned_path}")
        finetuned_state = _load_state_dict(finetuned_path, map_location="cpu")
        model.load_state_dict(finetuned_state, strict=False)

    model.to(device)
    return model


# Backward-compatible name used by older test scripts.
def load_pretrained_model(device="cpu"):
    return load_maia3_model(device=device)
