import sys
from pathlib import Path
from model_config import get_maia3_5m_config
import torch
from types import SimpleNamespace

from maia3.models import MAIA3Model
# cfg = get_maia3_5m_config()
# print(cfg.dim_vit)
# print(cfg.num_heads)
# print(cfg.head_hid_dim)
# model = MAIA3Model(cfg)
# print(model.token_projection)
sys.path.append(str(Path(__file__).resolve().parents[1]))



# model = MAIA3Model(cfg)

# checkpoint = torch.load(
#     Path.home()
#     / ".cache/huggingface/hub/models--UofTCSSLab--Maia3-5M/snapshots"
#     / "b6559de2398d7140b985f28fd2c19fb5e47ddabe"
#     / "maia3-5m.pt",
#     map_location="cpu",
# )
CHECKPOINT = (
    Path.home()
    / ".cache/huggingface/hub/models--UofTCSSLab--Maia3-5M/snapshots"
    / "b6559de2398d7140b985f28fd2c19fb5e47ddabe"
    / "maia3-5m.pt"
)

# # Maia3 checkpoints are raw state_dicts
# state = {k.replace("smolgen", "gab"): v for k, v in checkpoint.items()}

# missing, unexpected = model.load_state_dict(
#     state,
#     strict=False,
# )

# print("Missing:", len(missing))
# print("Unexpected:", len(unexpected))

# if missing:
#     print(missing)

# if unexpected:
#     print(unexpected)

# print()

# print("Example missing keys:")
# print(missing[:10])

# print()

# print("Example unexpected keys:")
# print(unexpected[:10])

# print()

# total = sum(p.numel() for p in model.parameters())
# loaded = 0

# for name, param in model.named_parameters():
#     if name in state:
#         loaded += param.numel()

# print(f"Loaded parameters: {loaded:,}/{total:,}")

def load_pretrained_model(device="cpu"):
    cfg = get_maia3_5m_config()

    model = MAIA3Model(cfg)

    checkpoint = torch.load(CHECKPOINT, map_location=device)

    state = {k.replace("smolgen", "gab"): v for k, v in checkpoint.items()}

    model.load_state_dict(state)

    model.to(device)

    return model