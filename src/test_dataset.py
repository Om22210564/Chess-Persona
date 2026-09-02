import numpy as np

from legacy.dataset import PGNParser


SAMPLE_PGN = """
[Event "Test"]
[Site "?"]
[Date "2024.01.01"]
[Round "1"]
[White "White"]
[Black "Black"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 1-0
"""


def test_legacy_pgn_parser_returns_training_samples(tmp_path):
    pgn_path = tmp_path / "sample.pgn"
    pgn_path.write_text(SAMPLE_PGN)

    parser = PGNParser(pgn_path, history_length=8)
    samples = parser.parse_games(max_games=1)

    assert len(samples) > 0

    history, policy, value = samples[0]
    history = np.stack(history)

    assert history.shape == (8, 17, 8, 8)
    assert isinstance(policy, int)
    assert value in {-1, 0, 1}
