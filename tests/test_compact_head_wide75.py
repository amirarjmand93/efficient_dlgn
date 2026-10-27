"""One focused construction/forward check for the separate wider head."""

from dataclasses import replace

import torch

from examples.phase_logic.config import ExperimentConfig
from examples.phase_logic.model import build_model_for_dataset, logic_layer_width_report


def test_wide75_preserves_baseline_and_cli_support():
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        kwargs = dict(device="cpu", dataset="cifar10", parametrization="compact",
                      compact_backend="torch", input_encoding="hybrid", hybrid_features="lbp",
                      input_thresholds=torch.full((3, 2), 0.5), num_quantile_bits=2,
                      num_sine_bits=0)
        torch.manual_seed(42)
        baseline = build_model_for_dataset(model_variant="build_model_cifar10_small_compact_head", **kwargs)
        torch.manual_seed(42)
        wide = build_model_for_dataset(model_variant="build_model_cifar10_small_compact_head_wide75", **kwargs)
        assert [baseline[i].out_dim for i in (6, 7)] == [92_400, 92_400]
        assert [wide[i].out_dim for i in (6, 7)] == [161_700, 161_700]
        assert [baseline[i].out_dim for i in range(1, 5)] == [128, 128, 512, 512]
        assert [wide[i].out_dim for i in range(1, 5)] == [224, 224, 896, 896]
        assert wide[1].in_shape == (12, 32, 32)
        assert wide[3].in_shape == (448, 16, 16)
        assert wide[4].in_shape == (896, 8, 8)
        assert wide[6].impl.in_dim == 114_688
        assert wide[7].impl.in_dim == 161_700
        assert wide[-1].k == 10 and wide[-1].tau == 55
        assert wide.concatenation_skips == baseline.concatenation_skips
        for key, value in baseline[0].state_dict().items():
            torch.testing.assert_close(wide[0].state_dict()[key], value, rtol=0, atol=0)
        assert sum(p.numel() for p in baseline.parameters()) == 2_977_280
        assert sum(p.numel() for p in wide.parameters()) == 5_210_240
        report = logic_layer_width_report(wide)
        assert [r["width"] for r in report] == [57344] * 4 + [161700] * 2
        assert sum(r["lut_node_applications"] for r in report) == 552_776
        with torch.no_grad():
            output = wide.eval()(torch.rand(1, 3, 32, 32))
        assert output.shape == (1, 10) and torch.isfinite(output).all()
        config = ExperimentConfig(dataset="cifar10", parametrization="compact", experiment="pure_ce",
                                  model_variant="build_model_cifar10_small_compact_head_wide75",
                                  iterations=700000)
        for changes in ({}, {"compact_layer_aware_resampling": True},
                        {"compact_efficient_layer_aware_resampling": True,
                         "compact_freeze_mode": "hybrid", "compact_dense_freeze": True}):
            replace(config, **changes).validate()
        assert config.uses_classrouted_compact_head
        assert "compact_shared_head2_wide75_v2_concat_c224x224x896x896_w161700x161700" in str(config.run_dir)
        print("wide75: parameters=5210240 widths=57344,57344,57344,57344,161700,161700 BOP/image=552776")
    finally:
        torch.set_num_threads(previous_threads)
