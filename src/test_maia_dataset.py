import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("maia3")

from maia_dataset import MaiaDataset, parse_time_control


SAMPLE_PGN = """
[Event "Test"]
[Site "?"]
[Date "2024.01.01"]
[Round "1"]
[White "White"]
[Black "Black"]
[WhiteElo "1600"]
[BlackElo "1400"]
[TimeControl "300+3"]
[Result "1-0"]

1. e4 { [%clk 0:04:59] } e5 { [%clk 0:04:58] } 2. Nf3 { [%clk 0:04:57] } Nc6 { [%clk 0:04:55] } 3. Bb5 { [%clk 0:04:54] } a6 { [%clk 0:04:50] } 1-0
"""


def test_parse_time_control():
    assert parse_time_control("300+3") == (300, 3)
    assert parse_time_control("60") == (60, 0)
    assert parse_time_control("-") == (0, 0)


def test_maia_dataset_returns_expected_tensor_shapes(tmp_path):
    pgn_path = tmp_path / "sample.pgn"
    pgn_path.write_text(SAMPLE_PGN)

    dataset = MaiaDataset(pgn_path)

    assert len(dataset) > 0

    tokens, policy, value = dataset[0]

    assert tokens.dtype == torch.float32
    assert tokens.shape == (64, 97)  # 12 piece planes * 8 history + 1 clk_ponder feature
    assert policy.ndim == 0
    assert value.item() in {0, 1, 2}


def test_maia_dataset_can_filter_to_user_moves_and_return_elos(tmp_path):
    pgn_path = tmp_path / "sample.pgn"
    pgn_path.write_text(SAMPLE_PGN)

    dataset = MaiaDataset(
        pgn_path,
        username="White",
        only_user_moves=True,
        include_elos=True,
    )

    assert len(dataset) == 3

    _, _, value, self_elo, oppo_elo = dataset[0]

    assert value.item() == 2
    assert self_elo.item() == 1600
    assert oppo_elo.item() == 1400
    assert dataset.stats["skipped_non_user_moves"] == 3
    assert dataset.stats["average_rating"] == 1600


def test_maia_dataset_supports_game_level_split(tmp_path):
    pgn_path = tmp_path / "sample.pgn"
    pgn_path.write_text(SAMPLE_PGN + "\n" + SAMPLE_PGN)

    all_dataset = MaiaDataset(pgn_path)
    train_dataset = MaiaDataset(pgn_path, split="train", val_fraction=0.5, split_seed=1)
    val_dataset = MaiaDataset(pgn_path, split="val", val_fraction=0.5, split_seed=1)

    assert len(train_dataset) + len(val_dataset) == len(all_dataset)
    assert train_dataset.stats["games"] + val_dataset.stats["games"] == 2


def test_maia_dataset_cache_round_trip(tmp_path):
    pgn_path = tmp_path / "sample.pgn"
    cache_path = tmp_path / "sample.cache.pt"
    pgn_path.write_text(SAMPLE_PGN)

    first = MaiaDataset(pgn_path, cache_path=cache_path)
    second = MaiaDataset(pgn_path, cache_path=cache_path)

    assert cache_path.exists()
    assert len(second) == len(first)
    assert second.stats == first.stats
