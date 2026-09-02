import pytest


torch = pytest.importorskip("torch")


def test_elo_tensors_match_batch_size():
    batch_size = 4

    self_elo = torch.full((batch_size,), 1500, dtype=torch.long)
    oppo_elo = torch.full((batch_size,), 1500, dtype=torch.long)

    assert self_elo.shape == (batch_size,)
    assert oppo_elo.shape == (batch_size,)
    assert self_elo.dtype == torch.long
    assert oppo_elo.dtype == torch.long
