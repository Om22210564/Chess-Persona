import chess


def mirror_square(square: str) -> str:
    """
    Mirror a square vertically.

    Example:
        a2 -> a7
        e4 -> e5
    """
    file = square[0]
    rank = str(9 - int(square[1]))
    return file + rank


def mirror_move(move: str) -> str:
    """
    Mirror a UCI move.

    Example:
        e2e4 -> e7e5
    """

    promotion = ""

    if len(move) > 4:
        promotion = move[4:]

    start = mirror_square(move[:2])
    end = mirror_square(move[2:4])

    return start + end + promotion


def generate_move_vocabulary():
    """
    Generate every possible move.

    Includes promotions.

    Returns:
        move_to_idx
        idx_to_move
    """

    moves = []

    # Normal moves
    for from_sq in chess.SQUARES:
        for to_sq in chess.SQUARES:

            move = (
                chess.square_name(from_sq)
                + chess.square_name(to_sq)
            )

            moves.append(move)

    # Promotions
    promotion_pieces = ["q", "r", "b", "n"]

    for file_from in "abcdefgh":
        for file_to in "abcdefgh":
            for piece in promotion_pieces:

                moves.append(
                    f"{file_from}7{file_to}8{piece}"
                )

    # Remove duplicates
    moves = sorted(list(set(moves)))

    move_to_idx = {
        move: idx
        for idx, move in enumerate(moves)
    }

    idx_to_move = {
        idx: move
        for move, idx in move_to_idx.items()
    }

    return move_to_idx, idx_to_move


if __name__ == "__main__":

    move_to_idx, idx_to_move = generate_move_vocabulary()

    print("Vocabulary size:", len(move_to_idx))

    print()

    print("First 20 moves:")

    for i in range(20):
        print(i, idx_to_move[i])