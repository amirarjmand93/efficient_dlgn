"""Focused checks for explicit no-freezing mode; no dataset needed."""

from dataclasses import replace

import pytest
import torch

from examples.phase_logic.compact import CandidateLogicLayer, CompactLogicController
from examples.phase_logic.run_experiment import build_parser, config_from_args


@pytest.mark.parametrize("mode", ["none", "entropy"])
def test_no_freeze_preserves_exploration_and_entropy_behavior(mode):
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        config = config_from_args(build_parser().parse_args([
            "--dataset", "cifar10", "--model-variant", "build_model_cifar10_small_compact_head_wide75",
            "--experiment", "pure_ce", "--parametrization", "compact", "--iterations", "700000",
            "--compact-freeze-mode", mode,
        ]))
        config = replace(config, compact_exploration_steps=2, compact_resample_patience=1,
                         compact_freeze_patience=1)
        config.validate()
        if mode == "none":
            assert "no_freeze_v1" in str(config.run_dir)
            with pytest.raises(ValueError, match="cannot be combined"):
                replace(config, compact_dense_freeze=True).validate()
        else:
            assert "no_freeze_v1" not in str(config.run_dir)
        layers = [CandidateLogicLayer(device="cpu", backend="torch", in_dim=32,
                                      out_dim=4, in_shape=(2, 4, 4)) for _ in range(4)]
        model = torch.nn.Sequential(*layers)
        with torch.no_grad():
            for layer in layers:
                layer.weights.zero_()
        controller = CompactLogicController(model, config)
        optimizer = torch.optim.Adam(model.parameters())
        controller.step(1, optimizer)
        controller.step(2, optimizer)
        assert controller.weak_rows == 16
        saved = {name: value.clone() for name, value in model.state_dict().items()}
        for step in range(3, 15):
            controller.step(step, optimizer)
        assert controller.weak_rows == 16  # Resampling stops at the original cutoff.
        assert sum(layer.is_frozen for layer in layers) == (0 if mode == "none" else 4)
        if mode == "none":
            assert all(layer.weights.requires_grad for layer in layers)
            assert controller.freeze_ema == controller.freeze_counters == {}
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value, saved[name], rtol=0, atol=0)
    finally:
        torch.set_num_threads(previous)
