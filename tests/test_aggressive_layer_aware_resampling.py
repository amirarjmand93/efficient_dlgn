"""CLI and layer policy wiring for the independent aggressive opcode mixture."""

from dataclasses import replace

import pytest
import torch

from examples.phase_logic.compact import CandidateLogicLayer, CompactLogicController
from examples.phase_logic.dense_freezing import enable_dense_group_freezing
from examples.phase_logic.hybrid_freezing import HybridFreezeController
from examples.phase_logic.run_experiment import build_parser, config_from_args


def test_aggressive_flag_is_separate_and_compatible_with_freezing_modes():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        args = ["--dataset", "cifar10", "--model-variant", "build_model_cifar10_small_compact_head_wide75",
                "--parametrization", "compact", "--experiment", "pure_ce", "--iterations", "700000"]
        parser = build_parser()
        baseline = config_from_args(parser.parse_args(args))
        config = config_from_args(parser.parse_args(args + ["--compact-aggressive-layer-aware-resampling"]))
        config.validate()
        assert not baseline.compact_aggressive_layer_aware_resampling
        assert not config.compact_efficient_layer_aware_resampling
        assert not config.compact_layer_aware_resampling
        assert "aggressive_layer_aware_resampling" not in str(baseline.run_dir)
        assert "aggressive_layer_aware_resampling_v1_conv25_d1_u50_i30_n5_c15_d2_u60_i25_n5_c10" in str(config.run_dir)
        for other in ("compact_efficient_layer_aware_resampling", "compact_layer_aware_resampling"):
            with pytest.raises(ValueError, match="only one"):
                replace(config, **{other: True}).validate()
        with pytest.raises(ValueError, match="four-convolution"):
            replace(config, model_variant="build_model_cifar10_small_mrlb_compact_head").validate()
        for mode in ("none", "entropy", "hybrid"):
            current = replace(config, compact_freeze_mode=mode, compact_dense_freeze=mode == "hybrid")
            current.validate()
            model = torch.nn.Sequential(*[
                CandidateLogicLayer(device="cpu", backend="torch", in_dim=32, out_dim=4,
                                    in_shape=(2, 4, 4)) for _ in range(4)
            ], *[CandidateLogicLayer(device="cpu", backend="torch", in_dim=8, out_dim=8)
                 for _ in range(2)])
            if current.compact_dense_freeze:
                enable_dense_group_freezing(model)
            before = {key: tensor.clone() for key, tensor in model.state_dict().items()}
            rng = torch.get_rng_state()
            controller = (HybridFreezeController if mode == "hybrid" else CompactLogicController)(model, current)
            assert [getattr(layer, "resample_preferred_ops", ()) for layer in model] == [
                (1, 7), (1, 7), (1, 7), (), (), (),
            ]
            assert [getattr(layer, "resample_efficient_dense", False) for layer in model] == [
                False, False, False, False, (0.30, 0.05, 0.15), (0.25, 0.05, 0.10),
            ]
            assert controller.freeze_enabled == (mode != "none")
            assert torch.equal(torch.get_rng_state(), rng)
            for key, tensor in model.state_dict().items():
                torch.testing.assert_close(tensor, before[key], rtol=0, atol=0)
    finally:
        torch.set_num_threads(previous)
