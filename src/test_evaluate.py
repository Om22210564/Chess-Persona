import pytest


pytest.importorskip("torch")
pytest.importorskip("maia3")

import chess

from evaluate import empty_stats, game_phase, policy_move_for_board, update_stats


def test_policy_move_for_board_mirrors_black_moves():
    board = chess.Board()
    assert policy_move_for_board(board, chess.Move.from_uci("e2e4")) == "e2e4"

    board.push(chess.Move.from_uci("e2e4"))
    assert policy_move_for_board(board, chess.Move.from_uci("e7e5")) == "e2e4"


def test_update_stats_tracks_top1_and_top5():
    stats = empty_stats()

    update_stats(stats, top1_ok=True, top5_ok=True)
    update_stats(stats, top1_ok=False, top5_ok=True)

    assert stats == {"total": 2, "top1": 1, "top5": 2}


def test_game_phase_labels_by_fullmove_number():
    board = chess.Board()
    assert game_phase(board) == "opening"

    board.fullmove_number = 20
    assert game_phase(board) == "middlegame"

    board.fullmove_number = 40
    assert game_phase(board) == "endgame"
