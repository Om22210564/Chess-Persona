import chess.pgn

from config import PGN_FILE


def count_games(pgn_path):
    """
    Counts the number of games in a PGN file.
    """
    total = 0

    with open(pgn_path, encoding="utf-8") as pgn:

        while True:

            game = chess.pgn.read_game(pgn)

            if game is None:
                break

            total += 1

    return total


def main():

    print(f"Reading: {PGN_FILE}")

    total = count_games(PGN_FILE)

    print(f"Total games found: {total}")


if __name__ == "__main__":
    main()