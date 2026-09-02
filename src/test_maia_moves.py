import chess
import pytest


pytest.importorskip("torch")
maia3_dataset = pytest.importorskip("maia3.dataset")
maia3_utils = pytest.importorskip("maia3.utils")


def test_mirror_move_for_black_perspective():
    assert maia3_utils.mirror_move("e7e5") == "e2e4"
    assert maia3_utils.mirror_move("a7a8q") == "a2a1q"


def test_legal_moves_mask_marks_starting_position_moves():
    all_moves = maia3_utils.get_all_possible_moves()
    move_to_index = {m: i for i, m in enumerate(all_moves)}

    board = chess.Board()
    mask = maia3_dataset.get_legal_moves_mask(board, move_to_index)

    assert mask.shape == (len(all_moves),)
    assert mask.sum().item() == board.legal_moves.count()
    assert mask[move_to_index["e2e4"]]
    assert not mask[move_to_index["e7e5"]]
