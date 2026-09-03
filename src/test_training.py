import argparse

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("maia3")

from train import configure_fine_tuning, topk_correct


def test_elo_tensors_match_batch_size():
    batch_size = 4

    self_elo = torch.full((batch_size,), 1500, dtype=torch.long)
    oppo_elo = torch.full((batch_size,), 1500, dtype=torch.long)

    assert self_elo.shape == (batch_size,)
    assert oppo_elo.shape == (batch_size,)
    assert self_elo.dtype == torch.long
    assert oppo_elo.dtype == torch.long


def test_topk_correct_counts_targets_in_top_k():
    logits = torch.tensor(
        [
            [0.1, 0.9, 0.0],
            [0.8, 0.1, 0.7],
        ]
    )
    targets = torch.tensor([1, 2])

    assert topk_correct(logits, targets, k=1) == 1
    assert topk_correct(logits, targets, k=2) == 2


class DummyBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj_sq_from = torch.nn.Linear(1, 1)
        self.proj_sq_to = torch.nn.Linear(1, 1)
        self.promo_bias_proj = torch.nn.Linear(1, 1)
        self.transformer = argparse.Namespace(
            layers=torch.nn.ModuleList([DummyBlock(), DummyBlock()])
        )


def test_configure_fine_tuning_policy_only():
    model = DummyModel()

    configure_fine_tuning(model, "policy")

    assert all(param.requires_grad for param in model.proj_sq_from.parameters())
    assert all(param.requires_grad for param in model.proj_sq_to.parameters())
    assert all(param.requires_grad for param in model.promo_bias_proj.parameters())
    assert not any(param.requires_grad for block in model.transformer.layers for param in block.parameters())


def test_configure_fine_tuning_last_block():
    model = DummyModel()

    configure_fine_tuning(model, "last-block")

    assert not next(model.transformer.layers[0].parameters()).requires_grad
    assert next(model.transformer.layers[1].parameters()).requires_grad
