import argparse
import sys
from collections import defaultdict, deque
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAIA3_ROOT = PROJECT_ROOT / "external" / "maia3"
if MAIA3_ROOT.exists() and str(MAIA3_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIA3_ROOT))

import chess
import chess.pgn
import torch

from load_pretrained import load_maia3_model
from model_config import get_maia3_5m_config

from maia3.dataset import (
    tokenize_board,
    get_historical_tokens,
    get_legal_moves_mask,
)
from maia3.utils import (
    get_all_possible_moves,
    mirror_move,
)


ALL_MOVES = get_all_possible_moves()
MOVE_TO_INDEX = {m: i for i, m in enumerate(ALL_MOVES)}
INDEX_TO_MOVE = {i: m for i, m in enumerate(ALL_MOVES)}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned Maia3 checkpoint on PGN moves.")
    parser.add_argument("--pgn-path", default="data/raw/sample.pgn")
    parser.add_argument("--model-path", default="best_policy.pt")
    parser.add_argument("--self-elo", type=int, default=1500)
    parser.add_argument("--oppo-elo", type=int, default=1500)
    parser.add_argument("--num-positions", type=int, default=20, help="Maximum positions to evaluate unless --all-positions is set.")
    parser.add_argument("--all-positions", action="store_true", help="Evaluate every supported move in the PGN.")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-position board output.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def empty_stats():
    return {"total": 0, "top1": 0, "top5": 0}


def update_stats(stats, top1_ok, top5_ok):
    stats["total"] += 1
    stats["top1"] += int(top1_ok)
    stats["top5"] += int(top5_ok)


def accuracy_line(label, stats):
    total = stats["total"]
    if total == 0:
        return f"{label}: no positions"
    return (
        f"{label}: {total} positions | "
        f"Top-1 {100 * stats['top1'] / total:.2f}% "
        f"({stats['top1']}/{total}) | "
        f"Top-5 {100 * stats['top5'] / total:.2f}% "
        f"({stats['top5']}/{total})"
    )


def game_phase(board):
    if board.fullmove_number <= 10:
        return "opening"
    if board.fullmove_number <= 30:
        return "middlegame"
    return "endgame"


def policy_move_for_board(board, move):
    move_uci = move.uci()
    return mirror_move(move_uci) if board.turn == chess.BLACK else move_uci


def display_position(board, actual_move, top_moves, top1_ok, top5_ok):
    print("=" * 60)
    print(board)
    print()
    print("FEN       :", board.fen())
    print("Actual    :", actual_move)
    print("Predicted :", top_moves[0] if top_moves else "<none>")
    print("Top-5     :", ", ".join(top_moves))
    print("Top-1 OK  :", "✓" if top1_ok else "✗")
    print("Top-5 OK  :", "✓" if top5_ok else "✗")
    print()


def print_grouped_stats(title, grouped_stats):
    print(title)
    for label in sorted(grouped_stats):
        print("  " + accuracy_line(str(label), grouped_stats[label]))


def evaluate(args):
    pgn_path = Path(args.pgn_path)
    if not pgn_path.exists():
        raise FileNotFoundError(f"PGN file not found: {pgn_path}")

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    cfg = get_maia3_5m_config()
    model = load_maia3_model(device=args.device, finetuned_path=model_path)
    model.eval()

    max_positions = None if args.all_positions else args.num_positions
    overall = empty_stats()
    by_color = defaultdict(empty_stats)
    by_opening = defaultdict(empty_stats)
    by_phase = defaultdict(empty_stats)
    skipped_unknown_moves = 0
    games = 0

    with open(pgn_path, encoding="utf-8") as f:
        while True:
            if max_positions is not None and overall["total"] >= max_positions:
                break

            game = chess.pgn.read_game(f)
            if game is None:
                break

            games += 1
            opening = game.headers.get("Opening", "Unknown")
            board = game.board()
            history = deque(maxlen=cfg.history)
            history.append(tokenize_board(board))

            for move in game.mainline_moves():
                policy_move = policy_move_for_board(board, move)

                if policy_move not in MOVE_TO_INDEX:
                    skipped_unknown_moves += 1
                    board.push(move)
                    history.append(tokenize_board(board))
                    continue

                tokens = get_historical_tokens(
                    history,
                    cfg,
                    base=0,
                    inc=0,
                    clk_left_before=0,
                    clk_ponder=0,
                ).unsqueeze(0).to(args.device)

                self_elo = torch.tensor([args.self_elo], dtype=torch.long, device=args.device)
                oppo_elo = torch.tensor([args.oppo_elo], dtype=torch.long, device=args.device)

                with torch.no_grad():
                    logits, _, _ = model(tokens, self_elo, oppo_elo)

                logits = logits[0]
                legal_mask = get_legal_moves_mask(board, MOVE_TO_INDEX).to(args.device)
                logits[~legal_mask] = -float("inf")

                top_indices = logits.topk(min(5, logits.numel())).indices.tolist()
                top_policy_moves = [INDEX_TO_MOVE[idx] for idx in top_indices]
                top_moves = [
                    mirror_move(move_uci) if board.turn == chess.BLACK else move_uci
                    for move_uci in top_policy_moves
                ]

                actual_move = move.uci()
                top1_ok = top_policy_moves[0] == policy_move
                top5_ok = policy_move in top_policy_moves

                update_stats(overall, top1_ok, top5_ok)
                update_stats(by_color["White" if board.turn == chess.WHITE else "Black"], top1_ok, top5_ok)
                update_stats(by_opening[opening], top1_ok, top5_ok)
                update_stats(by_phase[game_phase(board)], top1_ok, top5_ok)

                if not args.quiet:
                    display_position(board, actual_move, top_moves, top1_ok, top5_ok)

                board.push(move)
                history.append(tokenize_board(board))

                if max_positions is not None and overall["total"] >= max_positions:
                    break

    print("=" * 60)
    print(f"Games read          : {games}")
    print(f"Skipped unknown moves: {skipped_unknown_moves}")
    print(accuracy_line("Overall", overall))
    print()
    print_grouped_stats("By color", by_color)
    print()
    print_grouped_stats("By game phase", by_phase)
    print()
    print_grouped_stats("By opening", by_opening)


def main():
    evaluate(parse_args())


if __name__ == "__main__":
    main()
