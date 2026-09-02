import argparse
from collections import deque
from pathlib import Path

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
    parser.add_argument("--num-positions", type=int, default=20)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


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

    correct = 0
    total = 0
    shown = 0

    with open(pgn_path, encoding="utf-8") as f:
        while shown < args.num_positions:
            game = chess.pgn.read_game(f)
            if game is None:
                break

            board = game.board()
            history = deque(maxlen=cfg.history)
            history.append(tokenize_board(board))

            for move in game.mainline_moves():
                policy_move = move.uci()
                if board.turn == chess.BLACK:
                    policy_move = mirror_move(policy_move)

                if policy_move not in MOVE_TO_INDEX:
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

                pred_idx = torch.argmax(logits).item()
                pred_move = INDEX_TO_MOVE[pred_idx]

                if board.turn == chess.BLACK:
                    pred_move = mirror_move(pred_move)

                actual_move = move.uci()
                ok = pred_move == actual_move

                if ok:
                    correct += 1
                total += 1

                print("=" * 60)
                print(board)
                print()
                print("FEN      :", board.fen())
                print("Actual   :", actual_move)
                print("Predicted:", pred_move)
                print("Correct  :", "✓" if ok else "✗")
                print()

                shown += 1
                board.push(move)
                history.append(tokenize_board(board))

                if shown >= args.num_positions:
                    break

    print("=" * 60)
    if total:
        print(f"Accuracy : {correct}/{total} = {100 * correct / total:.2f}%")
    else:
        print("Accuracy : no evaluated positions")


def main():
    evaluate(parse_args())


if __name__ == "__main__":
    main()
