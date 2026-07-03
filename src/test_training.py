
import torch
from maia_dataset import MaiaDataset
import torch
from torch.utils.data import DataLoader

from maia_dataset import MaiaDataset
from load_pretrained import load_pretrained_model

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = MaiaDataset("data/raw/sample.pgn")
loader = DataLoader(dataset, batch_size=4)

model = load_pretrained_model()

tokens, policy, value = next(iter(loader))
self_elo = torch.full((4,), 1500, dtype=torch.long)
oppo_elo = torch.full((4,), 1500, dtype=torch.long)

logits, _, _ = model(tokens, self_elo, oppo_elo)

print(tokens.shape)
print(logits.shape)
print(policy.shape)