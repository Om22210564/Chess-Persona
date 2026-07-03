# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader

# from maia_dataset import MaiaDataset
# from load_pretrained import load_pretrained_model

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# BATCH_SIZE = 64
# EPOCHS = 5
# LR = 1e-5

# SELF_ELO = 1500
# OPPO_ELO = 1500

# dataset = MaiaDataset("data/train.pgn")

# loader = DataLoader(
#     dataset,
#     batch_size=BATCH_SIZE,
#     shuffle=True,
# )

# model = load_pretrained_model(DEVICE)

# criterion = nn.CrossEntropyLoss()

# optimizer = torch.optim.AdamW(
#     model.parameters(),
#     lr=LR,
# )

# model.train()

# for epoch in range(EPOCHS):

#     total_loss = 0

#     for tokens, policy, _ in loader:

#         tokens = tokens.to(DEVICE)
#         policy = policy.to(DEVICE)

#         batch = tokens.size(0)

#         self_elos = torch.full(
#             (batch,),
#             SELF_ELO,
#             dtype=torch.long,
#             device=DEVICE,
#         )

#         oppo_elos = torch.full(
#             (batch,),
#             OPPO_ELO,
#             dtype=torch.long,
#             device=DEVICE,
#         )

#         optimizer.zero_grad()

#         policy_logits, _, _ = model(
#             tokens,
#             self_elos,
#             oppo_elos,
#         )

#         loss = criterion(policy_logits, policy)

#         loss.backward()

#         optimizer.step()

#         total_loss += loss.item()

#     print(
#         f"Epoch {epoch+1}: "
#         f"{total_loss/len(loader):.4f}"
#     )

# torch.save(
#     model.state_dict(),
#     "maia3_finetuned.pt",
# )

# print("Training Finished")
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from maia_dataset import MaiaDataset

from maia3.models import MAIA3Model
from model_config import get_maia3_5m_config


# -----------------------------
# Config
# -----------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PGN_PATH = "data/raw/lichess_games.pgn"

BATCH_SIZE = 64
EPOCHS = 5
LR = 1e-4


# -----------------------------
# Dataset
# -----------------------------

dataset = MaiaDataset(PGN_PATH)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
)


# -----------------------------
# Model
# -----------------------------

cfg = get_maia3_5m_config()

model = MAIA3Model(cfg)

checkpoint = torch.load(
    Path.home()
    / ".cache/huggingface/hub/models--UofTCSSLab--Maia3-5M/snapshots"
    / "b6559de2398d7140b985f28fd2c19fb5e47ddabe"
    / "maia3-5m.pt",
    map_location="cpu",
)

state = {k.replace("smolgen", "gab"): v for k, v in checkpoint.items()}

model.load_state_dict(state, strict=False)
# Freeze all parameters
for param in model.parameters():
    param.requires_grad = False

# Train only the policy head
for param in model.proj_sq_from.parameters():
    param.requires_grad = True

for param in model.proj_sq_to.parameters():
    param.requires_grad = True

for param in model.promo_bias_proj.parameters():
    param.requires_grad = True

model.to(DEVICE)

print("Loaded pretrained Maia3-5M")


# -----------------------------
# Loss
# -----------------------------

criterion = nn.CrossEntropyLoss()


# -----------------------------
# Optimizer
# -----------------------------
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR,
    weight_decay=1e-4,
)
trainable = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

total = sum(
    p.numel()
    for p in model.parameters()
)

print(f"Trainable parameters: {trainable:,}")
print(f"Total parameters: {total:,}")
# -----------------------------
# Training
# -----------------------------

best_loss = float("inf")

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0
    correct = 0
    total = 0

    for tokens, policy, _ in loader:

        tokens = tokens.to(DEVICE)
        policy = policy.to(DEVICE)

        self_elo = torch.full(
            (tokens.size(0),),
            1500,
            dtype=torch.long,
            device=DEVICE,
        )

        oppo_elo = torch.full(
            (tokens.size(0),),
            1500,
            dtype=torch.long,
            device=DEVICE,
        )

        logits, _, _ = model(
            tokens,
            self_elo,
            oppo_elo,
        )

        loss = criterion(logits, policy)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # ---------- Accuracy ----------
        pred = logits.argmax(dim=1)

        correct += (pred == policy).sum().item()
        total += policy.size(0)

    avg_loss = running_loss / len(loader)
    accuracy = 100 * correct / total

    if avg_loss < best_loss:
        best_loss = avg_loss

        torch.save(
            model.state_dict(),
            "best_policy.pt",
        )

        print("Saved best model.")

    print(
        f"Epoch {epoch+1}/{EPOCHS} "
        f"Loss: {avg_loss:.4f} "
        f"Accuracy: {accuracy:.2f}%"
    )
torch.save(
    model.state_dict(),
    "maia3_finetuned_policy.pt",
)

print("Training complete.")