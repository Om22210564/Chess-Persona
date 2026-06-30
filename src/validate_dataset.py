from pathlib import Path

import pandas as pd

from config import PROCESSED_DATA


def print_section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def validate():

    games_path = PROCESSED_DATA / "games.csv"
    moves_path = PROCESSED_DATA / "moves.csv"

    games = pd.read_csv(games_path)
    moves = pd.read_csv(moves_path)

    # ---------------------------------------------------------
    print_section("DATASET OVERVIEW")

    print(f"Games               : {len(games)}")
    print(f"Moves               : {len(moves)}")
    print(f"Average Moves/Game  : {len(moves) / len(games):.2f}")

    # ---------------------------------------------------------
    print_section("GAME RESULTS")

    print(games["result"].value_counts())

    # ---------------------------------------------------------
    print_section("WHITE vs BLACK")

    print(games["color"].value_counts())

    # ---------------------------------------------------------
    print_section("TOP 10 OPENINGS")

    print(games["opening"].value_counts().head(10))

    # ---------------------------------------------------------
    print_section("PIECE USAGE")

    print(moves["piece"].value_counts())

    # ---------------------------------------------------------
    print_section("TACTICAL STATISTICS")

    total_moves = len(moves)

    capture_pct = 100 * moves["is_capture"].sum() / total_moves
    check_pct = 100 * moves["is_check"].sum() / total_moves
    castle_pct = 100 * moves["is_castling"].sum() / total_moves
    promotion_pct = 100 * moves["is_promotion"].sum() / total_moves
    en_passant_count = moves["is_en_passant"].sum()

    print(f"Captures      : {capture_pct:.2f}%")
    print(f"Checks        : {check_pct:.2f}%")
    print(f"Castling      : {castle_pct:.2f}%")
    print(f"Promotions    : {promotion_pct:.2f}%")
    print(f"En Passant    : {en_passant_count}")

    # ---------------------------------------------------------
    print_section("RATING")

    ratings = pd.to_numeric(moves["your_rating"])

    print(f"Lowest Rating : {ratings.min()}")
    print(f"Highest Rating: {ratings.max()}")
    print(f"Average Rating: {ratings.mean():.2f}")

    # ---------------------------------------------------------
    print_section("CLOCK")

    missing_clock = moves["clock"].isna().sum()

    print(f"Missing Clock Values : {missing_clock}")

    if missing_clock == 0:
        print(f"Average Clock Left   : {moves['clock'].mean():.2f} sec")

    # ---------------------------------------------------------
    print_section("MISSING VALUES")

    print(moves.isnull().sum())

    # ---------------------------------------------------------
    print_section("POSITIONS")

    unique_positions = moves["fen_before"].nunique()

    duplicate_positions = len(moves) - unique_positions

    print(f"Unique Positions     : {unique_positions}")
    print(f"Repeated Positions   : {duplicate_positions}")

    # ---------------------------------------------------------
    print_section("GAME LENGTH")

    moves_per_game = moves.groupby("game_id").size()

    print(f"Shortest Game : {moves_per_game.min()} of your moves")
    print(f"Longest Game  : {moves_per_game.max()} of your moves")
    print(f"Average       : {moves_per_game.mean():.2f}")

    # ---------------------------------------------------------
    print_section("VALIDATION")

    if len(games) == moves["game_id"].nunique():
        print("✓ Every game has move data.")
    else:
        print("✗ Some games may be missing.")

    print("Validation Complete.")


if __name__ == "__main__":
    validate()

""""
Below is the output of the script 

============================================================
DATASET OVERVIEW
============================================================
Games               : 304
Moves               : 10676
Average Moves/Game  : 35.12

============================================================
GAME RESULTS
============================================================
result
Win     156
Loss    133
Draw     15
Name: count, dtype: int64

============================================================
WHITE vs BLACK
============================================================
color
Black    154
White    150
Name: count, dtype: int64

============================================================
TOP 10 OPENINGS
============================================================
opening
Queen's Pawn Game: Accelerated London System    75
Sicilian Defense: Katalimov Variation           32
Queen's Pawn Game                               20
Indian Defense: Accelerated London System       11
Horwitz Defense                                 10
Benoni Defense: Old Benoni                      10
Sicilian Defense: Bowdler Attack                 9
Sicilian Defense: Smith-Morra Gambit             7
Englund Gambit                                   6
Sicilian Defense: Alapin Variation               6
Name: count, dtype: int64

============================================================
PIECE USAGE
============================================================
piece
Pawn      3173
King      1828
Bishop    1604
Knight    1542
Rook      1311
Queen     1218
Name: count, dtype: int64

============================================================
TACTICAL STATISTICS
============================================================
Captures      : 23.81%
Checks        : 8.52%
Castling      : 1.96%
Promotions    : 0.41%
En Passant    : 3

============================================================
RATING
============================================================
Lowest Rating : 728
Highest Rating: 1500
Average Rating: 1174.06

============================================================
CLOCK
============================================================
Missing Clock Values : 0
Average Clock Left   : 227.61 sec

============================================================
MISSING VALUES
============================================================
game_id            0
move_number        0
ply                0
color              0
fen_before         0
san                0
uci                0
clock              0
opening            0
time_control       0
result             0
your_rating        0
opponent_rating    0
piece              0
is_capture         0
is_check           0
is_castling        0
is_promotion       0
is_en_passant      0
piece_type         0
dtype: int64

============================================================
POSITIONS
============================================================
Unique Positions     : 9873
Repeated Positions   : 803

============================================================
GAME LENGTH
============================================================
Shortest Game : 2 of your moves
Longest Game  : 106 of your moves
Average       : 35.12

============================================================
VALIDATION
============================================================
✓ Every game has move data.
Validation Complete.
"""