import torch
from pathlib import Path

ckpt = torch.load(
    Path.home()
    / ".cache/huggingface/hub/models--UofTCSSLab--Maia3-5M/snapshots"
    / "b6559de2398d7140b985f28fd2c19fb5e47ddabe"
    / "maia3-5m.pt",
    map_location="cpu",
)

print(type(ckpt))

if isinstance(ckpt, dict):
    print("Keys:")
    for k in ckpt.keys():
        print(k)