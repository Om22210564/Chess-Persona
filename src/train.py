import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - convenience fallback
    tqdm = None

from load_pretrained import load_maia3_model
from maia_dataset import MaiaDataset


POLICY_MODULE_NAMES = ("proj_sq_from", "proj_sq_to", "promo_bias_proj")


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Maia3 policy layers on a PGN file.")
    parser.add_argument("--pgn-path", default="data/raw/lichess_games.pgn")
    parser.add_argument("--output-path", default="best_policy.pt")
    parser.add_argument("--final-output-path", default="maia3_finetuned_policy.pt")
    parser.add_argument("--metrics-path", default="training_metrics.jsonl")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4, help="Policy head learning rate.")
    parser.add_argument("--block-lr", type=float, default=1e-5, help="LR for unfrozen transformer blocks.")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--self-elo", type=int, default=1500)
    parser.add_argument("--oppo-elo", type=int, default=1500)
    parser.add_argument("--username", default=None)
    parser.add_argument("--only-user-moves", action="store_true")
    parser.add_argument("--use-pgn-elos", action="store_true")
    parser.add_argument("--log-dataset-stats", action="store_true")
    parser.add_argument("--split", choices=["all", "train", "val"], default="all")
    parser.add_argument("--val-fraction", type=float, default=0.0)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--fine-tune-mode", choices=["policy", "last-block", "last-two-blocks"], default="policy")
    parser.add_argument("--early-stopping-patience", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_fine_tuning(model, mode: str):
    """Freeze model, then unfreeze policy head and optionally final transformer blocks."""
    for param in model.parameters():
        param.requires_grad = False

    for name in POLICY_MODULE_NAMES:
        module = getattr(model, name)
        for param in module.parameters():
            param.requires_grad = True

    if mode in {"last-block", "last-two-blocks"}:
        blocks_to_unfreeze = 1 if mode == "last-block" else 2
        for block in model.transformer.layers[-blocks_to_unfreeze:]:
            for param in block.parameters():
                param.requires_grad = True


def create_optimizer(model, policy_lr, block_lr, weight_decay):
    policy_params = []
    policy_param_ids = set()
    for name in POLICY_MODULE_NAMES:
        for param in getattr(model, name).parameters():
            if param.requires_grad:
                policy_params.append(param)
                policy_param_ids.add(id(param))

    block_params = [
        param
        for param in model.parameters()
        if param.requires_grad and id(param) not in policy_param_ids
    ]

    param_groups = []
    if policy_params:
        param_groups.append({"params": policy_params, "lr": policy_lr})
    if block_params:
        param_groups.append({"params": block_params, "lr": block_lr})

    return torch.optim.AdamW(param_groups, weight_decay=weight_decay)


def build_dataset(args, split):
    return MaiaDataset(
        args.pgn_path,
        username=args.username,
        only_user_moves=args.only_user_moves,
        include_elos=args.use_pgn_elos,
        split=split,
        val_fraction=args.val_fraction,
        split_seed=args.split_seed,
        log_stats=args.log_dataset_stats,
    )


def unpack_batch(batch, args):
    if args.use_pgn_elos:
        tokens, policy, _, self_elo, oppo_elo = batch
        self_elo = self_elo.to(args.device)
        oppo_elo = oppo_elo.to(args.device)
    else:
        tokens, policy, _ = batch
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

    return tokens.to(args.device), policy.to(args.device), self_elo, oppo_elo


def topk_correct(logits, targets, k):
    k = min(k, logits.size(1))
    preds = logits.topk(k, dim=1).indices
    return preds.eq(targets.unsqueeze(1)).any(dim=1).sum().item()


def run_epoch(model, loader, criterion, args, optimizer=None, desc="train"):
    is_train = optimizer is not None
    model.train(is_train)

    running_loss = 0.0
    total = 0
    top1 = 0
    top5 = 0

    iterator = loader
    if tqdm is not None and not args.no_progress:
        iterator = tqdm(loader, desc=desc, leave=False)

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch in iterator:
            tokens, policy, self_elo, oppo_elo = unpack_batch(batch, args)

            logits, _, _ = model(tokens, self_elo, oppo_elo)
            loss = criterion(logits, policy)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            batch_size = policy.size(0)
            running_loss += loss.item() * batch_size
            total += batch_size
            top1 += (logits.argmax(dim=1) == policy).sum().item()
            top5 += topk_correct(logits, policy, k=5)

    return {
        "loss": running_loss / total if total else 0.0,
        "top1_acc": 100 * top1 / total if total else 0.0,
        "top5_acc": 100 * top5 / total if total else 0.0,
        "samples": total,
    }


def write_metrics(metrics_path, record):
    if not metrics_path:
        return
    metrics_path = Path(metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def train(args):
    set_seed(args.seed)

    pgn_path = Path(args.pgn_path)
    if not pgn_path.exists():
        raise FileNotFoundError(f"PGN file not found: {pgn_path}")

    if args.val_fraction > 0 and args.split == "all":
        train_dataset = build_dataset(args, split="train")
        val_dataset = build_dataset(args, split="val")
    else:
        train_dataset = build_dataset(args, split=args.split)
        val_dataset = None

    if len(train_dataset) == 0:
        raise ValueError(f"No training samples found in {pgn_path}")
    if val_dataset is not None and len(val_dataset) == 0:
        raise ValueError("Validation split produced no samples; lower --val-fraction or add more games.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )

    model = load_maia3_model(device=args.device)
    configure_fine_tuning(model, args.fine_tune_mode)

    print("Loaded pretrained Maia3-5M")
    print(f"Fine-tune mode: {args.fine_tune_mode}")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable:,}")
    print(f"Total parameters: {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer(model, args.lr, args.block_lr, args.weight_decay)

    best_metric = float("inf")
    epochs_without_improvement = 0

    if args.metrics_path:
        Path(args.metrics_path).unlink(missing_ok=True)
        write_metrics(
            args.metrics_path,
            {
                "type": "config",
                "args": vars(args),
                "train_samples": len(train_dataset),
                "val_samples": len(val_dataset) if val_dataset is not None else 0,
                "trainable_parameters": trainable,
                "total_parameters": total_params,
            },
        )

    for epoch in range(args.epochs):
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            args,
            optimizer=optimizer,
            desc=f"train {epoch + 1}/{args.epochs}",
        )

        val_metrics = None
        if val_loader is not None:
            val_metrics = run_epoch(
                model,
                val_loader,
                criterion,
                args,
                optimizer=None,
                desc=f"val {epoch + 1}/{args.epochs}",
            )

        checkpoint_metric = val_metrics["loss"] if val_metrics is not None else train_metrics["loss"]
        improved = checkpoint_metric < best_metric

        if improved:
            best_metric = checkpoint_metric
            epochs_without_improvement = 0
            Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), args.output_path)
            print(f"Saved best model to {args.output_path}")
        else:
            epochs_without_improvement += 1

        record = {
            "type": "epoch",
            "epoch": epoch + 1,
            "train": train_metrics,
            "val": val_metrics,
            "best_metric": best_metric,
            "fine_tune_mode": args.fine_tune_mode,
        }
        write_metrics(args.metrics_path, record)

        message = (
            f"Epoch {epoch + 1}/{args.epochs} "
            f"Train Loss: {train_metrics['loss']:.4f} "
            f"Top1: {train_metrics['top1_acc']:.2f}% "
            f"Top5: {train_metrics['top5_acc']:.2f}%"
        )
        if val_metrics is not None:
            message += (
                f" | Val Loss: {val_metrics['loss']:.4f} "
                f"Top1: {val_metrics['top1_acc']:.2f}% "
                f"Top5: {val_metrics['top5_acc']:.2f}%"
            )
        print(message)

        if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
            print(f"Early stopping after {epoch + 1} epochs.")
            break

    Path(args.final_output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.final_output_path)
    print(f"Training complete. Final model saved to {args.final_output_path}")


def main():
    train(parse_args())


if __name__ == "__main__":
    main()
