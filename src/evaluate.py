from pathlib import Path
from collections import deque

import chess
import chess.pgn
import torch

from maia3.models import MAIA3Model
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


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PGN_PATH = "data/raw/sample.pgn"          # <-- change if needed
MODEL_PATH = "best_policy.pt"

SELF_ELO = 1500
OPPO_ELO = 1500

NUM_POSITIONS = 20


###########################################################

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

finetuned = torch.load(
    MODEL_PATH,
    map_location="cpu",
)

model.load_state_dict(finetuned, strict=False)

model.to(DEVICE)
model.eval()

###########################################################

ALL_MOVES = get_all_possible_moves()
MOVE_TO_INDEX = {m: i for i, m in enumerate(ALL_MOVES)}
INDEX_TO_MOVE = {i: m for i, m in enumerate(ALL_MOVES)}

###########################################################

correct = 0
total = 0

with open(PGN_PATH) as f:

    shown = 0

    while shown < NUM_POSITIONS:

        game = chess.pgn.read_game(f)

        if game is None:
            break

        board = game.board()

        history = deque(maxlen=cfg.history)
        history.append(tokenize_board(board))

        for move in game.mainline_moves():

            if move.uci() not in MOVE_TO_INDEX:
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
            )

            tokens = tokens.unsqueeze(0).to(DEVICE)

            self_elo = torch.tensor(
                [SELF_ELO],
                dtype=torch.long,
                device=DEVICE,
            )

            oppo_elo = torch.tensor(
                [OPPO_ELO],
                dtype=torch.long,
                device=DEVICE,
            )

            with torch.no_grad():

                logits, _, _ = model(
                    tokens,
                    self_elo,
                    oppo_elo,
                )

            logits = logits[0]

            legal_mask = get_legal_moves_mask(
                board,
                MOVE_TO_INDEX,
            ).to(DEVICE)

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

            if shown >= NUM_POSITIONS:
                break

print("=" * 60)
print(f"Accuracy : {correct}/{total} = {100*correct/total:.2f}%")