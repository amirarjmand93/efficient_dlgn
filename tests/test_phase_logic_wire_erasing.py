import pytest
import torch

from torchlogix.layers import LogicDense

from examples.phase_logic.model import build_model_cifar10_smallmodel
from examples.phase_logic.wire_erasing import BooleanWireErasure


def test_wire_erasure_uses_explicit_enabled_flag():
    erasure = BooleanWireErasure(p=1.0)
    values = torch.tensor([[0.0, 0.25, 0.75, 1.0]])

    erasure.train()
    assert torch.equal(erasure(values), values)

    erasure.eval()
    erasure.set_enabled(True)
    assert torch.equal(erasure(values), torch.full_like(values, 0.5))


def test_wire_erasure_is_identity_in_export_mode():
    erasure = BooleanWireErasure(p=1.0)
    values = torch.tensor([[0.0, 1.0]])

    erasure.set_enabled(True)
    erasure.set_export_mode(True)
    erasure.set_enabled(True)

    assert erasure.enabled is False
    assert torch.equal(erasure(values), values)


@pytest.mark.parametrize("p", [-0.01, 1.01])
def test_wire_erasure_rejects_invalid_probability(p):
    with pytest.raises(ValueError, match="p must be in"):
        BooleanWireErasure(p=p)


def test_small_cifar_model_has_one_erasure_after_first_dense():
    thresholds = torch.tensor([[0.33, 0.66]] * 3)
    model = build_model_cifar10_smallmodel(
        torch.device("cpu"),
        parametrization="light",
        input_thresholds=thresholds,
        input_encoding="hybrid",
        num_quantile_bits=2,
        num_sine_bits=2,
    )

    erasure_indices = [
        index
        for index, module in enumerate(model)
        if isinstance(module, BooleanWireErasure)
    ]
    dense_indices = [
        index
        for index, module in enumerate(model)
        if isinstance(module, LogicDense)
    ]

    assert erasure_indices == [dense_indices[0] + 1]
    assert len(erasure_indices) == 1
    assert model[erasure_indices[0]].p == pytest.approx(0.0)
    assert model[erasure_indices[0]].enabled is False
