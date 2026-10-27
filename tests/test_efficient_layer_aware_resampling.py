"""Focused ELAR sampling and CLI checks; no dataset or training run needed."""

import copy
from dataclasses import replace

import pytest
import torch

from examples.phase_logic.compact import CandidateLogicLayer, CompactLogicController
from examples.phase_logic.dense_freezing import enable_dense_group_freezing
from examples.phase_logic.run_experiment import build_parser, config_from_args


@pytest.fixture(autouse=True, scope="module")
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("mixture", [(0.18, 0.08, 0.04), (0.12, 0.06, 0.02),
                                   (0.30, 0.05, 0.15), (0.25, 0.05, 0.10)])
def test_dense_mixture_preserves_connections_winners_and_optimizer_resets(mixture):
    torch.manual_seed(123)
    layer = CandidateLogicLayer(device="cpu", backend="torch", in_dim=8, out_dim=4096)
    layer.resample_efficient_dense = mixture
    with torch.no_grad():
        layer.weights.zero_()
        layer.weights[0, 11] = 20  # Strong winner must survive.
        layer.weights[-1, 3] = 3  # Selected intermediate-confidence row is unchanged.
    reference = copy.deepcopy(layer.impl)
    optimizer = torch.optim.Adam(layer.parameters(), amsgrad=True)
    keys = ("exp_avg", "exp_avg_sq", "max_exp_avg_sq")
    for key in keys:
        optimizer.state[layer.weights][key] = torch.ones_like(layer.weights)
    rows = torch.arange(layer.out_dim)
    rng = torch.get_rng_state()
    reference.resample_connections(rows, threshold_high=0.95, threshold_low=0.4,
                                   keep_strong_weights=False)
    torch.set_rng_state(rng)
    assert layer.resample(rows, optimizer) == (4094, 1)
    torch.testing.assert_close(layer.weights, reference.weights, rtol=0, atol=0)
    torch.testing.assert_close(layer.impl.indices, reference.indices, rtol=0, atol=0)
    assert layer.impl.gate_sequence[0, 11] == reference.gate_sequence[0, 11]
    torch.testing.assert_close(layer.impl.gate_sequence[-1], reference.gate_sequence[-1])
    for key in keys:
        assert torch.count_nonzero(optimizer.state[layer.weights][key][:-1]) == 0
        assert torch.all(optimizer.state[layer.weights][key][-1] == 1)
    codes = layer.impl.gate_sequence[1:-1].flatten().long()
    frequencies = codes.bincount(minlength=16).float() / codes.numel()
    expected = torch.full((16,), (1 - sum(mixture)) / 16)
    expected[[3, 5]] += mixture[0] / 2
    expected[[12, 10]] += mixture[1] / 2
    expected[[15, 0]] += mixture[2] / 2
    torch.testing.assert_close(frequencies, expected, rtol=0, atol=0.006)
    assert torch.all(frequencies > 0)


@pytest.mark.parametrize("group_freezing", [False, True])
def test_cli_and_layer_assignment(group_freezing):
    args = ["--dataset", "cifar10", "--model-variant", "build_model_cifar10_small_compact_head",
            "--parametrization", "compact", "--experiment", "pure_ce", "--iterations", "700000"]
    parser = build_parser()
    config = config_from_args(parser.parse_args(args + ["--compact-efficient-layer-aware-resampling"]))
    config.validate()
    assert not config.compact_layer_aware_resampling
    assert "efficient_layer_aware_resampling_v2_conv25_dense1_u70_i18_n8_c4_dense2_u80_i12_n6_c2" in str(config.run_dir)
    with pytest.raises(ValueError, match="only one"):
        replace(config, compact_layer_aware_resampling=True).validate()
    with pytest.raises(ValueError, match="four-convolution"):
        replace(config, model_variant="build_model_cifar10_small_mrlb_compact_head").validate()
    model = torch.nn.Sequential(*[
        CandidateLogicLayer(device="cpu", backend="torch", in_dim=32, out_dim=4,
                            in_shape=(2, 4, 4)) for _ in range(4)
    ], *[CandidateLogicLayer(device="cpu", backend="torch", in_dim=8, out_dim=8)
         for _ in range(2)])
    if group_freezing:
        enable_dense_group_freezing(model)
    state = copy.deepcopy(model.state_dict())
    rng = torch.get_rng_state()
    CompactLogicController(model, config)
    assert [getattr(layer, "resample_preferred_ops", ()) for layer in model] == [
        (1, 7), (1, 7), (1, 7), (), (), (),
    ]
    assert [getattr(layer, "resample_efficient_dense", False) for layer in model] == [
        False, False, False, False, (0.18, 0.08, 0.04), (0.12, 0.06, 0.02),
    ]
    assert torch.equal(torch.get_rng_state(), rng)
    for key, value in state.items():
        torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
