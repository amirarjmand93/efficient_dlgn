"""Focused checks for opt-in CIFAR-10 loss-probed rewiring."""

import json

import pytest
import torch

from examples.phase_logic.compare_rewiring import compare_runs
from examples.phase_logic.config import ExperimentConfig
from examples.phase_logic.rewiring import LossProbedRewirer
from torchlogix.layers import ClassRoutedLogicDense


def _routed_layer(in_dim=8):
    return ClassRoutedLogicDense(
        in_dim=in_dim,
        out_dim=2 * in_dim,
        num_classes=2,
        shared_input=True,
        shared_input_subset=False,
        lut_rank=2,
        parametrization="light",
        parametrization_kwargs={"forward_sampling": "soft", "temperature": 1.0},
        device="cpu",
    )


def test_candidate_pairs_screen_full_shared_input_without_self_pairs():
    layer = _routed_layer(in_dim=12)
    inputs = torch.zeros(6, 12)
    inputs[:, 11] = torch.tensor([0., 0., 0., 1., 1., 1.])
    gradient = torch.tensor([-1., -1., -1., 1., 1., 1.])

    pairs = layer.candidate_pairs(
        inputs, gradient, 0, feature_pool_size=12, num_candidates=3
    )

    assert pairs.shape == (3, 2)
    assert (pairs[:, 0] != pairs[:, 1]).all()
    assert (pairs == 11).any(dim=1).all()


def test_row_priority_avoids_helpful_gates():
    output = torch.tensor([
        [0., 0., 0.], [0., 0., 0.], [1., 1., 0.], [1., 1., 0.],
    ])
    gradient = torch.tensor([
        [-1., 1., 1.], [-1., 1., -1.],
        [1., -1., 1.], [1., -1., -1.],
    ])
    priority = LossProbedRewirer._row_priority(output, gradient)
    assert priority[0] > priority[2] > priority[1]


def test_entropy_monitor_waits_for_warmup_then_detects_stalled_rows():
    layer = _routed_layer()
    model = torch.nn.Sequential(layer, torch.nn.Linear(layer.out_dim, 2))
    with torch.no_grad():
        layer.weight.zero_()  # maximum Bernoulli entropy, not converged
    config = ExperimentConfig(
        dataset="cifar10",
        model_variant="build_model_cifar10_small_classrouted",
        rewire_loss_probe=True,
        rewire_warmup=2,
        rewire_interval=2,
        rewire_entropy_threshold=0.1,
        rewire_entropy_delta=0.001,
        rewire_max_rows=1,
    )
    rewirer = LossProbedRewirer(model, config)
    optimizer = torch.optim.Adam(model.parameters())
    inputs = torch.rand(4, 8)
    targets = torch.tensor([0, 1, 0, 1])

    probe = dict(probe_inputs=inputs.flip(0), probe_targets=targets.flip(0))
    assert rewirer.maybe_rewire(model, inputs, targets, 1, optimizer, **probe) is None
    first = rewirer.maybe_rewire(model, inputs, targets, 2, optimizer, **probe)
    assert first["converged_rows"] == 0
    assert first["stalled_rows"] == 0
    assert first["attempted_rows"] == 0
    second = rewirer.maybe_rewire(model, inputs, targets, 4, optimizer, **probe)
    assert second["stalled_rows"] == layer.out_dim
    assert second["attempted_rows"] == 1


def test_loss_probe_rewires_only_improving_row_and_resets_its_adam_moments():
    layer = _routed_layer()
    classifier = torch.nn.Linear(layer.out_dim, 2, bias=False)
    model = torch.nn.Sequential(layer, classifier)
    model.train()
    with torch.no_grad():
        layer.weight.fill_(-10.0)
        layer.connections.indices[:, 0] = torch.tensor([2, 3])
        classifier.weight.zero_()
        classifier.weight[0, 0] = 8.0

    config = ExperimentConfig(
        dataset="cifar10",
        model_variant="build_model_cifar10_small_classrouted",
        rewire_loss_probe=True,
        iterations=10,
        temperature_warmup=0,
        phase_warmup=0,
        rewire_warmup=1,
        rewire_interval=1,
        rewire_max_rows=1,
        rewire_candidates=1,
        rewire_feature_pool=8,
        rewire_entropy_threshold=1.0,
    )
    config.validate()
    rewirer = LossProbedRewirer(model, config)
    layer.candidate_pairs = lambda *args, **kwargs: torch.tensor([[0, 1]])
    new_weight = torch.tensor([-10.0, -10.0, 10.0, 10.0])
    layer.parametrization.init_weights = lambda count, device: new_weight[None].repeat(count, 1)
    optimizer = torch.optim.Adam(model.parameters())
    optimizer.state[layer.weight] = {
        "exp_avg": torch.ones_like(layer.weight),
        "exp_avg_sq": torch.ones_like(layer.weight),
    }
    inputs = torch.zeros(6, 8)
    inputs[:, 0] = torch.tensor([0., 0., 0., 1., 1., 1.])
    targets = torch.tensor([1, 1, 1, 0, 0, 0])
    other_indices = layer.connections.indices[:, 1:].clone()
    other_weights = layer.weight[1:].clone()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    before = torch.nn.functional.cross_entropy(model(inputs), targets)

    event = rewirer.maybe_rewire(
        model, inputs, targets, step=1, optimizer=optimizer,
        probe_inputs=inputs.flip(0), probe_targets=targets.flip(0),
    )

    assert event["attempted_rows"] == 1
    assert event["accepted_rows"] == 1
    assert torch.equal(layer.connections.indices[:, 0], torch.tensor([0, 1]))
    assert torch.equal(layer.weight[0], new_weight)
    assert torch.equal(layer.connections.indices[:, 1:], other_indices)
    assert torch.equal(layer.weight[1:], other_weights)
    assert torch.count_nonzero(optimizer.state[layer.weight]["exp_avg"][0]) == 0
    assert torch.count_nonzero(optimizer.state[layer.weight]["exp_avg_sq"][0]) == 0
    assert torch.all(optimizer.state[layer.weight]["exp_avg"][1:] == 1)
    assert torch.all(optimizer.state[layer.weight]["exp_avg_sq"][1:] == 1)
    assert torch.nn.functional.cross_entropy(model(inputs), targets) < before
    assert sum(parameter.numel() for parameter in model.parameters()) == parameter_count


def test_search_batch_gain_is_rejected_when_separate_probe_batch_worsens():
    layer = _routed_layer()
    classifier = torch.nn.Linear(layer.out_dim, 2, bias=False)
    model = torch.nn.Sequential(layer, classifier)
    with torch.no_grad():
        layer.weight.fill_(-10.0)
        layer.connections.indices[:, 0] = torch.tensor([2, 3])
        classifier.weight.zero_()
        classifier.weight[0, 0] = 8.0
    layer.candidate_pairs = lambda *args, **kwargs: torch.tensor([[0, 1]])
    new_weight = torch.tensor([-10.0, -10.0, 10.0, 10.0])
    layer.parametrization.init_weights = lambda count, device: new_weight[None].repeat(count, 1)
    config = ExperimentConfig(
        dataset="cifar10",
        model_variant="build_model_cifar10_small_classrouted",
        rewire_loss_probe=True,
        rewire_warmup=1,
        rewire_interval=1,
        rewire_max_rows=1,
        rewire_candidates=1,
        rewire_entropy_threshold=1.0,
    )
    inputs = torch.zeros(6, 8)
    inputs[:, 0] = torch.tensor([0., 0., 0., 1., 1., 1.])
    targets = torch.tensor([1, 1, 1, 0, 0, 0])
    original_indices = layer.connections.indices.clone()
    original_weights = layer.weight.detach().clone()
    event = LossProbedRewirer(model, config).maybe_rewire(
        model, inputs, targets, 1, torch.optim.Adam(model.parameters()),
        probe_inputs=inputs.flip(0), probe_targets=(1 - targets).flip(0),
    )
    assert event["attempted_rows"] == 1
    assert event["accepted_rows"] == 0
    assert torch.equal(layer.connections.indices, original_indices)
    assert torch.equal(layer.weight, original_weights)


def test_non_improving_candidate_preserves_routing_weights_and_moments():
    layer = _routed_layer()
    classifier = torch.nn.Linear(layer.out_dim, 2, bias=False)
    model = torch.nn.Sequential(layer, classifier)
    model.train()
    with torch.no_grad():
        layer.weight.fill_(-10.0)
        layer.weight[0] = torch.tensor([-10.0, -10.0, 10.0, 10.0])
        layer.connections.indices[:, 0] = torch.tensor([0, 1])
        classifier.weight.zero_()
        classifier.weight[0, 0] = 8.0
    config = ExperimentConfig(
        dataset="cifar10",
        model_variant="build_model_cifar10_small_classrouted",
        rewire_loss_probe=True,
        rewire_warmup=1,
        rewire_interval=1,
        rewire_max_rows=1,
        rewire_candidates=1,
        rewire_entropy_threshold=1.0,
    )
    rewirer = LossProbedRewirer(model, config)
    layer.candidate_pairs = lambda *args, **kwargs: torch.tensor([[2, 3]])
    layer.parametrization.init_weights = lambda count, device: torch.full(
        (count, 4), -10.0
    )
    optimizer = torch.optim.Adam(model.parameters())
    optimizer.state[layer.weight] = {
        "exp_avg": torch.ones_like(layer.weight),
        "exp_avg_sq": torch.ones_like(layer.weight),
    }
    inputs = torch.zeros(6, 8)
    inputs[:, 0] = torch.tensor([0., 0., 0., 1., 1., 1.])
    targets = torch.tensor([1, 1, 1, 0, 0, 0])
    indices_before = layer.connections.indices.clone()
    weights_before = layer.weight.detach().clone()

    event = rewirer.maybe_rewire(
        model, inputs, targets, 1, optimizer,
        probe_inputs=inputs.flip(0), probe_targets=targets.flip(0),
    )

    assert event["accepted_rows"] == 0
    assert torch.equal(layer.connections.indices, indices_before)
    assert torch.equal(layer.weight, weights_before)
    assert torch.all(optimizer.state[layer.weight]["exp_avg"] == 1)
    assert torch.all(optimizer.state[layer.weight]["exp_avg_sq"] == 1)


def test_fixed_baseline_keeps_its_existing_result_directory():
    baseline = ExperimentConfig(
        dataset="cifar10",
        model_variant="build_model_cifar10_small_classrouted",
    )
    rewired = ExperimentConfig(
        dataset="cifar10",
        model_variant="build_model_cifar10_small_classrouted",
        rewire_loss_probe=True,
    )
    assert "rewire_loss_probe" not in str(baseline.run_dir)
    assert "rewire_loss_probe_v2" in str(rewired.run_dir)


def test_rewiring_is_cifar10_only_and_rejects_phase_loss():
    common = dict(rewire_loss_probe=True, iterations=10, temperature_warmup=0,
                  phase_warmup=0, rewire_warmup=1)
    ExperimentConfig(
        dataset="cifar10", model_variant="build_model_cifar10_small_classrouted",
        experiment="annealing", **common,
    ).validate()
    with pytest.raises(ValueError, match="CIFAR-10"):
        ExperimentConfig(
            dataset="cifar100", model_variant="build_model_cifar100_small_classrouted",
            **common,
        ).validate()
    with pytest.raises(ValueError, match="Allen-Cahn"):
        ExperimentConfig(
            dataset="cifar10", model_variant="build_model_cifar10_small_classrouted",
            experiment="annealing_phase", **common,
        ).validate()


def test_rewirer_handles_a_sequential_subclass_without_slice_constructor():
    class CustomSequential(torch.nn.Sequential):
        def __init__(self, required_name, *modules):
            self.required_name = required_name
            super().__init__(*modules)

    layer = _routed_layer()
    model = CustomSequential("custom", layer, torch.nn.Linear(layer.out_dim, 2))
    config = ExperimentConfig(
        dataset="cifar10",
        model_variant="build_model_cifar10_small_classrouted",
        rewire_loss_probe=True,
        rewire_warmup=1,
        rewire_interval=1,
        rewire_entropy_threshold=1.0,
        rewire_max_rows=1,
    )
    event = LossProbedRewirer(model, config).maybe_rewire(
        model, torch.rand(4, 8), torch.tensor([0, 1, 0, 1]),
        1, torch.optim.Adam(model.parameters()),
        probe_inputs=torch.rand(4, 8), probe_targets=torch.tensor([1, 0, 1, 0]),
    )
    assert event["attempted_rows"] == 1


def test_comparison_checks_matched_seed_and_training_settings(tmp_path):
    fixed_root = tmp_path / "fixed"
    rewired_root = tmp_path / "rewired"
    for root, enabled, accuracy in (
        (fixed_root, False, 0.30),
        (rewired_root, True, 0.32),
    ):
        run_dir = root / "seed_42"
        run_dir.mkdir(parents=True)
        config = ExperimentConfig(
            dataset="cifar10",
            model_variant="build_model_cifar10_small_classrouted",
            seed=42,
            rewire_loss_probe=enabled,
        ).to_dict()
        (run_dir / "config.json").write_text(json.dumps(config))
        (run_dir / "summary.json").write_text(json.dumps({
            "final": {
                "test_acc_discrete": accuracy,
                "test_acc_relaxed": accuracy + 0.01,
            },
            "training_elapsed_seconds": 10.0,
            "rewiring": {"elapsed_seconds": 1.0, "accepted_rows": 2},
        }))

    rows = compare_runs(fixed_root, rewired_root, [42])
    assert rows[0]["rewired_discrete"] - rows[0]["fixed_discrete"] == pytest.approx(0.02)

    mismatched = json.loads((rewired_root / "seed_42" / "config.json").read_text())
    mismatched["batch_size"] = 64
    (rewired_root / "seed_42" / "config.json").write_text(json.dumps(mismatched))
    with pytest.raises(ValueError, match="batch_size"):
        compare_runs(fixed_root, rewired_root, [42])
