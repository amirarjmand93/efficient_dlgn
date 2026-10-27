"""Focused checks for the opt-in opcode prior, without full training runs."""

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


@pytest.mark.parametrize("preferred", [(), (1, 7), (6, 9)])
def test_opcode_prior_preserves_routing_winners_and_adam_resets(preferred):
    torch.manual_seed(42)
    layer = CandidateLogicLayer(device="cpu", backend="torch", in_dim=8, out_dim=4096)
    layer.resample_preferred_ops = preferred
    optimizer = torch.optim.Adam(layer.parameters(), amsgrad=True)
    with torch.no_grad():
        layer.weights.zero_()  # Weak rows: replace all candidates.
        layer.weights[0, 11] = 20  # Strong: preserve this non-preferred winner.
        layer.weights[-1, 3] = 3  # Intermediate confidence: leave untouched.
    for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
        optimizer.state[layer.weights][key] = torch.ones_like(layer.weights)
    original = copy.deepcopy(layer.impl)
    reference = copy.deepcopy(layer.impl)
    rows = torch.arange(layer.out_dim - 1)  # Last row is not selected.
    rng = torch.get_rng_state()
    reference.resample_connections(rows, threshold_high=0.95, threshold_low=0.4,
                                   keep_strong_weights=False)
    reference_rng = torch.get_rng_state()
    torch.set_rng_state(rng)
    assert layer.resample(rows, optimizer) == (4094, 1)
    torch.testing.assert_close(layer.weights, reference.weights, rtol=0, atol=0)
    torch.testing.assert_close(layer.impl.indices, reference.indices, rtol=0, atol=0)
    assert layer.impl.gate_sequence[0, 11] == original.gate_sequence[0, 11]
    torch.testing.assert_close(layer.impl.indices[:, 0, 11], original.indices[:, 0, 11])
    torch.testing.assert_close(layer.impl.gate_sequence[-1], original.gate_sequence[-1])
    for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
        assert torch.count_nonzero(optimizer.state[layer.weights][key][:-1]) == 0
        assert torch.all(optimizer.state[layer.weights][key][-1] == 1)
    if preferred:
        codes = layer.impl.gate_sequence[1:-1].flatten().long()
        frequencies = codes.bincount(minlength=16).float() / codes.numel()
        # Each preferred opcode: .75/16 + .25/2; all others: .75/16.
        expected = torch.full((16,), 0.75 / 16)
        expected[list(preferred)] += 0.25 / 2
        torch.testing.assert_close(frequencies, expected, rtol=0, atol=0.01)
        assert torch.all(frequencies > 0)
    else:
        torch.testing.assert_close(layer.impl.gate_sequence, reference.gate_sequence)
        assert torch.equal(torch.get_rng_state(), reference_rng)


def test_cli_policy_mapping_dense_wrapper_and_separate_run_path():
    parser = build_parser()
    args = ["--dataset", "cifar10", "--model-variant", "build_model_cifar10_small_compact_head",
            "--parametrization", "compact", "--experiment", "pure_ce",
            "--iterations", "700000"]
    baseline = config_from_args(parser.parse_args(args))
    config = config_from_args(parser.parse_args(args + ["--compact-layer-aware-resampling"]))
    config.validate()
    assert not baseline.compact_layer_aware_resampling
    assert "layer_aware_resampling" not in str(baseline.run_dir)
    assert "layer_aware_resampling_v1_preferred25" in str(config.run_dir)
    with pytest.raises(ValueError, match="four-convolution"):
        replace(config, model_variant="build_model_cifar10_compact_sixconv").validate()
    model = torch.nn.Sequential(*[
        CandidateLogicLayer(device="cpu", backend="torch", in_dim=32, out_dim=4,
                            in_shape=(2, 4, 4)) for _ in range(4)
    ], *[
        CandidateLogicLayer(device="cpu", backend="torch", in_dim=8, out_dim=8)
        for _ in range(2)
    ])
    enable_dense_group_freezing(model)
    before = copy.deepcopy(model.state_dict())
    rng = torch.get_rng_state()
    CompactLogicController(model, config)
    assert [getattr(layer, "resample_preferred_ops", ()) for layer in model] == [
        (1, 7), (1, 7), (1, 7), (), (), (6, 9),
    ]
    assert torch.equal(torch.get_rng_state(), rng)
    for key, value in before.items():
        torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
