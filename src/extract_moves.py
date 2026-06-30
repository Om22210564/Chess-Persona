import chess
import chess.pgn
import pandas as pd

from config import PGN_FILE, PROCESSED_DATA, USERNAME
from utils import extract_clock, get_piece_info


def extract_moves():

    rows = []

    with open(PGN_FILE, encoding="utf-8") as pgn:

        while True:

            game = chess.pgn.read_game(pgn)

            if game is None:
                break

            headers = game.headers

            game_id = headers["GameId"]
            opening = headers.get("Opening")
            time_control = headers.get("TimeControl")

            if headers["White"] == USERNAME:
                my_color = chess.WHITE
                result = (
                    "Win" if headers["Result"] == "1-0"
                    else "Loss" if headers["Result"] == "0-1"
                    else "Draw"
                )

                your_rating = headers.get("WhiteElo")
                opponent_rating = headers.get("BlackElo")

            else:

                my_color = chess.BLACK

                result = (
                    "Win" if headers["Result"] == "0-1"
                    else "Loss" if headers["Result"] == "1-0"
                    else "Draw"
                )

                your_rating = headers.get("BlackElo")
                opponent_rating = headers.get("WhiteElo")

            board = game.board()

            node = game

            ply = 0

            while node.variations:

                next_node = node.variation(0)

                move = next_node.move

                ply += 1
                                    

                if board.turn == my_color:

                    fen_before = board.fen()

                    san = board.san(move)

                    uci = move.uci()

                    clock = extract_clock(next_node.comment)

                    is_capture = board.is_capture(move)

                    is_castling = board.is_castling(move)

                    is_en_passant = board.is_en_passant(move)

                    is_promotion = move.promotion is not None

                    piece_type, piece = get_piece_info(board, move)

                    # Temporarily make the move
                    board.push(move)

                    is_check = board.is_check()

                    board.pop()

                    rows.append(
                        {
                            "game_id": game_id,
                            "move_number": board.fullmove_number,
                            "ply": ply,
                            "color": "White" if my_color else "Black",
                            "fen_before": fen_before,
                            "san": san,
                            "uci": uci,
                            "clock": clock,
                            "opening": opening,
                            "time_control": time_control,
                            "result": result,
                            "your_rating": your_rating,
                            "opponent_rating": opponent_rating,
                            "is_capture": is_capture,
                            "is_check": is_check,
                            "is_castling": is_castling,
                            "is_promotion": is_promotion,
                            "is_en_passant": is_en_passant,
                            "piece_type": piece_type,
                            "piece": piece,
                            
                        }
                    )


                board.push(move)

                node = next_node

    df = pd.DataFrame(rows)

    PROCESSED_DATA.mkdir(parents=True, exist_ok=True)

    output = PROCESSED_DATA / "moves.csv"

    df.to_csv(output, index=False)

    print(df.head())

    print()

    print(f"Total moves extracted : {len(df)}")

    print(f"Saved to : {output}")


if __name__ == "__main__":
    extract_moves()