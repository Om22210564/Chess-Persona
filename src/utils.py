import re
import chess

def clock_to_seconds(clock_str):
    """
    Converts a clock string like '0:03:00' to seconds.
    Returns None if clock_str is None.
    """
    if clock_str is None:
        return None

    h, m, s = map(int, clock_str.split(":"))
    return h * 3600 + m * 60 + s


def extract_clock(comment):
    """
    Extract clock from a PGN comment.

    Example:
    '{ [%clk 0:02:58] }'
        -> 178
    """
    if not comment:
        return None

    match = re.search(r"%clk\s+(\d+:\d+:\d+)", comment)

    if match:
        return clock_to_seconds(match.group(1))

    return None


def piece_name(board, move): # WIll remove this function as the get_piece_info function also does the same task
    """
    Returns the name of the piece making the move.
    """
    piece = board.piece_at(move.from_square)

    if piece is None:
        return None

    names = {
        chess.PAWN: "Pawn",
        chess.KNIGHT: "Knight",
        chess.BISHOP: "Bishop",
        chess.ROOK: "Rook",
        chess.QUEEN: "Queen",
        chess.KING: "King",
    }

    return names[piece.piece_type]

def get_piece_info(board, move):

    piece = board.piece_at(move.from_square)

    if piece is None:
        return None, None

    names = {
        chess.PAWN: "Pawn",
        chess.KNIGHT: "Knight",
        chess.BISHOP: "Bishop",
        chess.ROOK: "Rook",
        chess.QUEEN: "Queen",
        chess.KING: "King",
    }

    return piece.piece_type, names[piece.piece_type]