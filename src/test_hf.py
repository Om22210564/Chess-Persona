import torch

from load_pretrained import download_maia3_checkpoint


ckpt = torch.load(
    download_maia3_checkpoint(),
    map_location="cpu",
)

print(type(ckpt))

if isinstance(ckpt, dict):
    print("Keys:")
    for k in ckpt.keys():
        print(k)
