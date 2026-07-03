import torch

from model import ChessTransformer

model = ChessTransformer()

dummy = torch.randn(
    4,
    8,
    17,
    8,
    8,
)

policy, value = model(dummy)

print("Policy:", policy.shape)
print("Value :", value.shape)

print()

total_params = sum(p.numel() for p in model.parameters())

print(f"Total Parameters: {total_params:,}")