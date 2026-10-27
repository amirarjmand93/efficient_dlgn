"""Focused semantic checks for the optional CompactLogic CIFAR-10 variant."""

from dataclasses import replace

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from examples.phase_logic.compact import (
    CandidateLogicLayer,
    CompactLogicController,
    UniformThermometerEncoder,
    _boolean_operation,
)
from examples.phase_logic.config import ExperimentConfig
from examples.phase_logic.run_experiment import build_parser, config_from_args


@pytest.fixture(autouse=True, scope="module")
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def test_gate_truth_tables_and_uniform_threshold_channel_order():
    codes = torch.arange(16)
    for a, b, bit in ((0, 0, 3), (0, 1, 2), (1, 0, 1), (1, 1, 0)):
        actual = _boolean_operation(torch.tensor(float(a)), torch.tensor(float(b)), codes)
        torch.testing.assert_close(actual, ((codes >> bit) & 1).float())
    image = torch.tensor([0.25, 0.5, 0.75]).reshape(1, 3, 1, 1)
    torch.testing.assert_close(UniformThermometerEncoder(3)(image).flatten(),
                               torch.tensor([0., 1., 1., 0., 0., 1., 0., 0., 0.]))


def test_resampling_preserves_winner_and_resets_only_changed_adam_rows():
    layer = CandidateLogicLayer(device="cpu", backend="torch", in_dim=8, out_dim=8)
    optimizer = torch.optim.Adam(layer.parameters(), amsgrad=True)
    layer(torch.rand(3, 8)).sum().backward()
    optimizer.step()
    with torch.no_grad():
        layer.weights[0].zero_()
        layer.weights[0, 3] = 20
        layer.weights[1].zero_()
    weights = layer.weights.detach().clone()
    indices = layer.impl.indices.clone()
    gates = layer.impl.gate_sequence.clone()
    for name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
        optimizer.state[layer.weights][name].fill_(1)
    assert layer.resample(torch.tensor([0, 1]), optimizer) == (1, 1)
    torch.testing.assert_close(layer.weights[2:], weights[2:])
    torch.testing.assert_close(layer.impl.indices[:, 2:], indices[:, 2:])
    torch.testing.assert_close(layer.impl.indices[:, 0, 3], indices[:, 0, 3])
    assert layer.impl.gate_sequence[0, 3] == gates[0, 3]
    torch.testing.assert_close(layer.weights[0].softmax(-1)[3], torch.tensor(0.9))
    for name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
        state = optimizer.state[layer.weights][name]
        assert torch.count_nonzero(state[:2]) == 0
        assert torch.all(state[2:] == 1)


@pytest.mark.parametrize("stride", [1, 2])
def test_conv_spatial_alignment_gradients_and_single_channel_routing(stride):
    layer = CandidateLogicLayer(device="cpu", backend="torch", in_dim=50,
                                out_dim=4, in_shape=(2, 5, 5), stride=stride)
    x = torch.rand(2, 2, 5, 5, requires_grad=True)
    layer(x).square().sum().backward()
    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(layer.weights.grad).all()
    assert torch.all(layer.impl.offsets_ch == (torch.arange(4) % 2)[None, :, None])
    layer.eval()
    result = layer(x.detach())
    expected = torch.empty_like(result)
    # Residual initialization selects gate ID 3: output is the first operand.
    for channel in range(4):
        input_channel = int(layer.impl.offsets_ch[0, channel, 3])
        dh = int(layer.impl.offsets_h[0, channel, 3])
        dw = int(layer.impl.offsets_w[0, channel, 3])
        for h in range(result.shape[2]):
            for w in range(result.shape[3]):
                ih, iw = h * stride + dh, w * stride + dw
                expected[:, channel, h, w] = x.detach()[:, input_channel, ih, iw] if 0 <= ih < 5 and 0 <= iw < 5 else 0
    torch.testing.assert_close(result, expected)


def test_controller_phase_boundary_and_frozen_checkpoint_roundtrip():
    layers = [CandidateLogicLayer(device="cpu", backend="torch", in_dim=32,
                                  out_dim=4, in_shape=(2, 4, 4)) for _ in range(4)]
    model = torch.nn.Sequential(*layers)
    config = ExperimentConfig(compact_exploration_steps=2, compact_resample_patience=1,
                              compact_freeze_patience=1)
    controller = CompactLogicController(model, config)
    optimizer = torch.optim.Adam(model.parameters())
    with torch.no_grad():
        for layer in layers:
            layer.weights.zero_()
    controller.step(1, optimizer)
    controller.step(2, optimizer)
    assert controller.weak_rows == 16
    for step in range(3, 11):
        controller.step(step, optimizer)
    assert len(controller.summary()["frozen_conv_layers"]) == 4
    assert controller.weak_rows == 16
    restored = torch.nn.Sequential(*[
        CandidateLogicLayer(device="cpu", backend="torch", in_dim=32,
                            out_dim=4, in_shape=(2, 4, 4)) for _ in range(4)
    ])
    restored.load_state_dict(model.state_dict())
    restored.train()
    assert all(layer.is_frozen and not layer.weights.requires_grad for layer in restored)
    image = torch.rand(1, 2, 4, 4)
    training_output = restored[0].train()(image)
    torch.testing.assert_close(training_output, restored[0].eval()(image))
    other = CompactLogicController(restored, config)
    other.load_state_dict(controller.state_dict())
    assert other.summary() == controller.summary()


def test_small_model_cli_and_short_training(tmp_path, monkeypatch):
    from examples.phase_logic import training

    args = build_parser().parse_args([
        "--dataset", "cifar10", "--model-variant", "build_model_cifar10_compact_small",
        "--parametrization", "compact", "--experiment", "pure_ce",
        "--input-encoding", "thermometer", "--num-input-bits", "3",
        "--iterations", "3", "--temperature-warmup", "0", "--phase-warmup", "0",
        "--compact-backend", "torch", "--compact-exploration-steps", "1",
        "--compact-freeze-patience", "1", "--device", "cpu", "--save-model",
    ])
    config = replace(config_from_args(args), output_dir=tmp_path, eval_frequency=3, batch_size=2)
    config.validate()
    with pytest.raises(ValueError, match="parametrization compact"):
        replace(config, parametrization="light").validate()
    images = torch.rand(2, 3, 32, 32)
    loader = DataLoader(TensorDataset(images, torch.tensor([0, 1])), batch_size=2)
    monkeypatch.setattr(training, "build_data_loaders", lambda _: (loader, loader, loader))
    summary = training.run_experiment(config)
    assert summary["parameter_count"] == 2_580_480
    assert [layer["width"] for layer in summary["logic_layer_widths"]] == [32768] * 4 + [80000] * 2
    assert len(summary["compact"]["frozen_conv_layers"]) == 1
    assert (config.run_dir / "compact.csv").exists()
    assert (config.run_dir / "final_model.pt").exists()
