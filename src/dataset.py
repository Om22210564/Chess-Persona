# Not needed replaced by maia_dataset.py


import chess
import chess.pgn

from collections import deque

from board_encoder import encode_board
from move_vocab import generate_move_vocabulary, mirror_move

move_to_idx, idx_to_move = generate_move_vocabulary()


class PGNParser:

    def __init__(self, pgn_file, history_length=8):

        self.pgn_file = pgn_file
        self.history_length = history_length

    def parse_games(self, max_games=None):

        samples = []

        with open(self.pgn_file, encoding="utf-8") as f:

            game_counter = 0

            while True:

                game = chess.pgn.read_game(f)

                if game is None:
                    break

                board = game.board()

                history = deque(maxlen=self.history_length)

                result = game.headers["Result"]

                if result == "1-0":
                    value = 1
                elif result == "0-1":
                    value = -1
                else:
                    value = 0

                for move in game.mainline_moves():

                    current_board = encode_board(board)

                    history.append(current_board)

                    if len(history) < self.history_length:
                        board.push(move)
                        continue

                    if board.turn == chess.BLACK:
                        move_uci = mirror_move(move.uci())
                    else:
                        move_uci = move.uci()

                    if move_uci not in move_to_idx:
                        board.push(move)
                        continue

                    samples.append(
                        (
                            list(history),
                            move_to_idx[move_uci],
                            value,
                        )
                    )

                    board.push(move)

                game_counter += 1

                if max_games and game_counter >= max_games:
                    break

        return samples