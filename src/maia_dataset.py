from collections import deque

import chess
import chess.pgn
import torch
from torch.utils.data import Dataset
from maia3.utils import mirror_move
from maia3.dataset import (
    tokenize_board,
    get_historical_tokens,
)

from maia3.utils import (
    get_all_possible_moves,
)


ALL_MOVES = get_all_possible_moves()
MOVE_TO_INDEX = {m: i for i, m in enumerate(ALL_MOVES)}


class MaiaDataset(Dataset):

    def __init__(self, pgn_path, history=8):

        self.history = history
        self.samples = []

        with open(pgn_path) as f:

            while True:

                game = chess.pgn.read_game(f)

                if game is None:
                    break

                board = game.board()

                history_queue = deque(maxlen=history)

                history_queue.append(tokenize_board(board))

                result = game.headers.get("Result", "*")

                if result == "1-0":
                    value = 2
                elif result == "0-1":
                    value = 0
                else:
                    value = 1

                for move in game.mainline_moves():

                    move_uci = move.uci()

                    # Maia3 always predicts from White's perspective
                    if board.turn == chess.BLACK:
                        move_uci = mirror_move(move_uci)

                    if move_uci not in MOVE_TO_INDEX:
                        board.push(move)
                        history_queue.append(tokenize_board(board))
                        continue

                    tokens = get_historical_tokens(
                        history_queue,
                        type(
                            "cfg",
                            (),
                            {
                                "history": history,
                                "include_time_info": False,
                            },
                        ),
                        base=0,
                        inc=0,
                        clk_left_before=0,
                        clk_ponder=0,
                    )

                    self.samples.append(
                        (
                            tokens,
                            MOVE_TO_INDEX[move_uci],
                            value,
                        )
                    )

                    board.push(move)
                    history_queue.append(tokenize_board(board))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        x, policy, value = self.samples[idx]

        return (
            x.float(),
            torch.tensor(policy),
            torch.tensor(value),
        )