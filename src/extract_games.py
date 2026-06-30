import chess.pgn
import pandas as pd

from config import PGN_FILE, PROCESSED_DATA, USERNAME


def get_result(game, username):
    headers = game.headers

    white = headers["White"]
    black = headers["Black"]
    result = headers["Result"]

    if username == white:
        if result == "1-0":
            return "Win"
        elif result == "0-1":
            return "Loss"
        else:
            return "Draw"

    if username == black:
        if result == "0-1":
            return "Win"
        elif result == "1-0":
            return "Loss"
        else:
            return "Draw"

    return "Unknown"


def extract_games():

    games = []

    with open(PGN_FILE, encoding="utf-8") as pgn:

        while True:

            game = chess.pgn.read_game(pgn)

            if game is None:
                break

            h = game.headers

            if h["White"] == USERNAME:
                color = "White"
                opponent = h["Black"]
                your_rating = h.get("WhiteElo")
                opponent_rating = h.get("BlackElo")
                rating_diff = h.get("WhiteRatingDiff")

            else:
                color = "Black"
                opponent = h["White"]
                your_rating = h.get("BlackElo")
                opponent_rating = h.get("WhiteElo")
                rating_diff = h.get("BlackRatingDiff")

            games.append({
                "game_id": h.get("GameId"),
                "date": h.get("Date"),
                "color": color,
                "opponent": opponent,
                "your_rating": your_rating,
                "opponent_rating": opponent_rating,
                "rating_diff": rating_diff,
                "result": get_result(game, USERNAME),
                "opening": h.get("Opening"),
                "eco": h.get("ECO"),
                "termination": h.get("Termination"),
                "time_control": h.get("TimeControl"),
                # "variant": h.get("Variant"),
            })

    df = pd.DataFrame(games)

    PROCESSED_DATA.mkdir(parents=True, exist_ok=True)

    output = PROCESSED_DATA / "games.csv"

    df.to_csv(output, index=False)

    print(df.head())

    print(f"\nSaved {len(df)} games to {output}")


if __name__ == "__main__":
    extract_games()