"""Routing and integration checks for the separate candidate-head variant."""

from dataclasses import replace

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from examples.phase_logic.classrouted_compact import BitConcatCompactModel, ClassRoutedCandidateDense
from examples.phase_logic.compact import CandidateLogicLayer
from examples.phase_logic.config import ExperimentConfig
from examples.phase_logic.model import (
    build_model_cifar10_compact_small,
    build_model_cifar10_small_classrouted_compact_head,
    logic_layer_width_report,
)
from torchlogix.layers import ClassRoutedLogicDense, LogicConv2d, OrPooling2d


@pytest.mark.parametrize("shared", [True, False])
def test_candidate_routes_preserve_access_and_resampling_bounds(shared):
    original = ClassRoutedLogicDense(
        in_dim=4 if shared else 12, out_dim=12 if shared else 6,
        num_classes=3, shared_input=shared, parametrization="light", device="cpu",
    )
    head = ClassRoutedCandidateDense(original, device="cpu", backend="torch")
    x = torch.rand(3, original.in_dim, requires_grad=True)
    head(x)[:, :head.out_per_class].sum().backward()
    assert torch.isfinite(x.grad).all()
    if shared:
        assert torch.all(x.grad.abs().sum(0) > 0)
    else:
        assert torch.count_nonzero(x.grad[:, head.in_per_class:]) == 0
    optimizer = torch.optim.Adam(head.parameters())
    optimizer.step()
    for class_index, route in enumerate(head.routes):
        source_pairs = original.connections.indices[:,
            class_index * head.out_per_class:(class_index + 1) * head.out_per_class]
        offset = 0 if shared else class_index * head.in_per_class
        torch.testing.assert_close(route.impl.indices[:, :, 3], source_pairs - offset)
        with torch.no_grad():
            route.weights.zero_()
        route.resample(torch.arange(head.out_per_class), optimizer)
        assert torch.all(route.impl.indices >= 0)
        assert torch.all(route.impl.indices < head.in_per_class)
    # Replacements remain class-local because each engine receives only its route.
    x.grad = None
    head(x)[:, :head.out_per_class].sum().backward()
    if not shared:
        assert torch.count_nonzero(x.grad[:, head.in_per_class:]) == 0


def test_wide_shared_input_uses_full_representation_without_slicing():
    original = ClassRoutedLogicDense(
        in_dim=32, out_dim=24, num_classes=3, shared_input=True,
        shared_input_subset=True, parametrization="light", device="cpu",
    )
    head = ClassRoutedCandidateDense(original, device="cpu", backend="torch")
    assert head.pooled_shared_engine
    assert len(head.routes) == 1
    route = head.routes[0]
    assert route.impl.in_dim == 32
    assert route.impl.indices.shape == (2, 24, 16)
    # All three class blocks can read the last shared feature.
    rows = torch.tensor([0, 8, 16])
    with torch.no_grad():
        route.weights.zero_()
        route.weights[:, 3] = 20
        route.impl.indices[0, rows, 3] = 31
        route.impl.indices[1, rows, 3] = 30
    x = torch.rand(2, 32, requires_grad=True)
    output = head(x)
    torch.testing.assert_close(output[:, rows], x[:, 31:32].expand(-1, 3))
    output[:, rows].sum().backward()
    assert torch.all(x.grad[:, 31] > 0)


def test_concat_skips_keep_gradients_when_transform_branches_block_them():
    class Zero(torch.nn.Module):
        def forward(self, x):
            return x * 0

    model = BitConcatCompactModel(
        torch.nn.Identity(), torch.nn.Identity(), Zero(), torch.nn.Identity(),
        Zero(), torch.nn.Flatten(), torch.nn.Identity(), torch.nn.Identity(),
        torch.nn.Identity(),
    )
    x = torch.rand(1, 1, 2, 2, requires_grad=True)
    model(x).sum().backward()
    torch.testing.assert_close(x.grad, torch.ones_like(x))


def test_new_variant_uses_concat_skips_and_fully_shared_wide_head():
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        kwargs = dict(input_encoding="hybrid", num_quantile_bits=4, num_sine_bits=4,
                      input_thresholds=torch.tensor([[0.2, 0.4, 0.6, 0.8]]).repeat(3, 1))
        torch.manual_seed(42421015)
        baseline = build_model_cifar10_compact_small(torch.device("cpu"), compact_backend="torch", **kwargs)
        torch.manual_seed(42421015)
        candidate = build_model_cifar10_small_classrouted_compact_head(
            torch.device("cpu"), compact_backend="torch", **kwargs,
        )
        assert sum(p.numel() for p in candidate.parameters()) == 2_580_480
        assert sum(p.numel() for p in candidate.parameters()) == sum(p.numel() for p in baseline.parameters())
        assert [row["width"] for row in logic_layer_width_report(candidate)] == [
            32768, 32768, 32768, 32768, 80000, 80000]
        assert sum(row["lut_node_applications"] for row in logic_layer_width_report(candidate)) == 291_072
        assert not any(isinstance(layer, (LogicConv2d, OrPooling2d)) for layer in candidate.modules())
        for index in (1, 2, 4):
            old, new = baseline[index], candidate[index]
            assert isinstance(new, CandidateLogicLayer) and new.is_convolution
            for key, value in old.state_dict().items():
                torch.testing.assert_close(value, new.state_dict()[key], rtol=0, atol=0)
        assert candidate[3].in_shape == (256, 16, 16)
        # The existing single-channel kernel assignment covers both source banks.
        assert torch.unique(candidate[3].impl.offsets_ch).numel() == 256
        assert not any(isinstance(layer, ClassRoutedCandidateDense) for layer in candidate.modules())
        for head, in_dim in zip((candidate[6], candidate[7]), (65536, 80000)):
            assert isinstance(head, CandidateLogicLayer) and not head.is_convolution
            assert head.impl.in_dim == in_dim and head.out_dim == 80000
            assert torch.all(head.impl.indices >= 0)
            assert torch.all(head.impl.indices < in_dim)
            assert torch.unique(head.impl.indices).numel() == in_dim
        image = torch.rand(1, 3, 32, 32)
        torch.testing.assert_close(candidate[0](image), baseline[0](image))
        captured = {}
        handles = [candidate[index].register_forward_hook(
            lambda module, inputs, output, index=index: captured.__setitem__(index, output)
        ) for index in range(1, 5)]
        handles.extend(candidate[index].register_forward_pre_hook(
            lambda module, inputs, index=index: captured.__setitem__(f"input{index}", inputs[0])
        ) for index in (3, 6))
        try:
            with torch.no_grad():
                scores = candidate.eval()(image)
            assert scores.shape == (1, 10) and torch.isfinite(scores).all()
            torch.testing.assert_close(captured["input3"], torch.cat((captured[1], captured[2]), dim=1))
            torch.testing.assert_close(captured["input6"], torch.cat((captured[3], captured[4]), dim=1).flatten(1))
            assert torch.all((captured["input6"] == 0) | (captured["input6"] == 1))
        finally:
            for handle in handles:
                handle.remove()
        assert candidate[-1].k == baseline[-1].k == 10
        assert candidate[-1].tau == baseline[-1].tau == 40
    finally:
        torch.set_num_threads(previous_threads)


def test_cuda_head_smoke_training(tmp_path, monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    pytest.importorskip("compactlogic")
    from examples.phase_logic import training

    config = ExperimentConfig(
        dataset="cifar10", model_variant="build_model_cifar10_small_classrouted_compact_head",
        parametrization="compact", input_encoding="hybrid", num_quantile_bits=4,
        num_sine_bits=4, experiment="pure_ce", device="cuda", compact_backend="cuda",
        iterations=5, eval_frequency=5, compact_exploration_steps=1,
        compact_freeze_patience=1, learning_rate=1e-8,
        temperature_warmup=0, phase_warmup=0, output_dir=tmp_path,
        batch_size=2, save_model=True,
    )
    config.validate()
    with pytest.raises(ValueError):
        replace(config, rewire_loss_probe=True).validate()
    with pytest.raises(ValueError, match="pure_ce"):
        replace(config, experiment="annealing").validate()
    loader = DataLoader(TensorDataset(torch.rand(2, 3, 32, 32), torch.tensor([0, 1])), batch_size=2)
    monkeypatch.setattr(training, "build_data_loaders", lambda _: (loader, loader, loader))
    monkeypatch.setattr(training, "build_cifar10_input_thresholds", lambda *_: torch.tensor(
        [[0.2, 0.4, 0.6, 0.8]]).repeat(3, 1))
    summary = training.run_experiment(config)
    assert summary["parameter_count"] == 2_580_480
    assert len(summary["logic_layer_widths"]) == 6
    assert summary["compact"]["frozen_conv_layers"] == ["1", "2"]
    assert (config.run_dir / "best_model.pt").is_file()
    assert (config.run_dir / "compact.csv").is_file()
    assert "compact_shared_head_v5_concat_w80000x80000_" in str(config.run_dir)
