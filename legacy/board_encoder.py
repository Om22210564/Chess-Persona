import chess
import numpy as np


PIECE_TO_PLANE = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5,
}


def encode_board(board: chess.Board):
    """
    Encode a chess.Board into Maia-style planes.

    Returns
    -------
    np.ndarray
        Shape (17, 8, 8)
    """

    planes = np.zeros((17, 8, 8), dtype=np.float32)

    ##########################################
    # Piece planes
    ##########################################

    for square in chess.SQUARES:

        piece = board.piece_at(square)

        if piece is None:
            continue

        row = 7 - chess.square_rank(square)
        col = chess.square_file(square)

        plane = PIECE_TO_PLANE[piece.piece_type]

        if piece.color == chess.BLACK:
            plane += 6

        planes[plane, row, col] = 1

    ##########################################
    # Side to move
    ##########################################

    if board.turn == chess.WHITE:
        planes[12] = 1

    ##########################################
    # Castling rights
    ##########################################

    if board.has_kingside_castling_rights(chess.WHITE):
        planes[13] = 1

    if board.has_queenside_castling_rights(chess.WHITE):
        planes[14] = 1

    if board.has_kingside_castling_rights(chess.BLACK):
        planes[15] = 1

    if board.has_queenside_castling_rights(chess.BLACK):
        planes[16] = 1

    return planes