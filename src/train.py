import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from load_pretrained import load_maia3_model
from maia_dataset import MaiaDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Maia3 policy layers on a PGN file.")
    parser.add_argument("--pgn-path", default="data/raw/lichess_games.pgn")
    parser.add_argument("--output-path", default="best_policy.pt")
    parser.add_argument("--final-output-path", default="maia3_finetuned_policy.pt")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--self-elo", type=int, default=1500)
    parser.add_argument("--oppo-elo", type=int, default=1500)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def freeze_for_policy_head_training(model):
    """Freeze the Maia3 backbone and train only policy projection layers."""
    for param in model.parameters():
        param.requires_grad = False

    for module in (model.proj_sq_from, model.proj_sq_to, model.promo_bias_proj):
        for param in module.parameters():
            param.requires_grad = True


def train(args):
    pgn_path = Path(args.pgn_path)
    if not pgn_path.exists():
        raise FileNotFoundError(f"PGN file not found: {pgn_path}")

    dataset = MaiaDataset(pgn_path)
    if len(dataset) == 0:
        raise ValueError(f"No training samples found in {pgn_path}")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    model = load_maia3_model(device=args.device)
    freeze_for_policy_head_training(model)

    print("Loaded pretrained Maia3-5M")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable:,}")
    print(f"Total parameters: {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_loss = float("inf")

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for tokens, policy, _ in loader:
            tokens = tokens.to(args.device)
            policy = policy.to(args.device)

            self_elo = torch.full(
                (tokens.size(0),),
                args.self_elo,
                dtype=torch.long,
                device=args.device,
            )
            oppo_elo = torch.full(
                (tokens.size(0),),
                args.oppo_elo,
                dtype=torch.long,
                device=args.device,
            )

            logits, _, _ = model(tokens, self_elo, oppo_elo)
            loss = criterion(logits, policy)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += (pred == policy).sum().item()
            total += policy.size(0)

        avg_loss = running_loss / len(loader)
        accuracy = 100 * correct / total if total else 0.0

        if avg_loss < best_loss:
            best_loss = avg_loss
            Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), args.output_path)
            print(f"Saved best model to {args.output_path}")

        print(
            f"Epoch {epoch + 1}/{args.epochs} "
            f"Loss: {avg_loss:.4f} "
            f"Accuracy: {accuracy:.2f}%"
        )

    Path(args.final_output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.final_output_path)
    print(f"Training complete. Final model saved to {args.final_output_path}")


def main():
    train(parse_args())


if __name__ == "__main__":
    main()
