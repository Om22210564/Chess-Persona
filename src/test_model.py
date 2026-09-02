import pytest


torch = pytest.importorskip("torch")

from legacy.model import ChessTransformer


def test_legacy_chess_transformer_output_shapes():
    model = ChessTransformer()
    dummy = torch.randn(4, 8, 17, 8, 8)

    policy, value = model(dummy)

    assert policy.shape == (4, 4160)
    assert value.shape == (4, 1)
