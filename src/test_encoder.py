import chess

from legacy.board_encoder import encode_board


def test_starting_board_encoder_shape_and_counts():
    board = chess.Board()
    tensor = encode_board(board)

    assert tensor.shape == (17, 8, 8)
    assert tensor.dtype.name == "float32"

    assert tensor[0].sum() == 8  # white pawns
    assert tensor[6].sum() == 8  # black pawns
    assert tensor[5].sum() == 1  # white king
    assert tensor[11].sum() == 1  # black king


def test_starting_board_metadata_planes():
    board = chess.Board()
    tensor = encode_board(board)

    assert tensor[12][0][0] == 1  # white to move
    assert tensor[13][0][0] == 1  # white kingside castling
    assert tensor[14][0][0] == 1  # white queenside castling
    assert tensor[15][0][0] == 1  # black kingside castling
    assert tensor[16][0][0] == 1  # black queenside castling
