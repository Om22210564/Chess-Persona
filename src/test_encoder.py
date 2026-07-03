import chess

from board_encoder import encode_board

board = chess.Board()

tensor = encode_board(board)

print("Shape:", tensor.shape)

print("White pawns:", tensor[0].sum())

print("Black pawns:", tensor[6].sum())

print("White king:", tensor[5].sum())

print("Black king:", tensor[11].sum())

print("Side to move:", tensor[12][0][0])

print("White kingside:", tensor[13][0][0])

print("White queenside:", tensor[14][0][0])

print("Black kingside:", tensor[15][0][0])

print("Black queenside:", tensor[16][0][0])