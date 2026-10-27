"""Focused opt-in freezing and matched-budget six-convolution checks."""

from dataclasses import replace

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from examples.phase_logic.compact import CandidateLogicLayer
from examples.phase_logic.config import ExperimentConfig
from examples.phase_logic.hybrid_freezing import HybridFreezeController
from examples.phase_logic.model import build_model_for_dataset, logic_layer_width_report
from examples.phase_logic.run_experiment import build_parser, config_from_args


@pytest.fixture(autouse=True, scope="module")
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def toy_model():
    layers = [CandidateLogicLayer(device="cpu", backend="torch", in_dim=4,
                                 out_dim=2, in_shape=(1, 2, 2)),
              CandidateLogicLayer(device="cpu", backend="torch", in_dim=8,
                                 out_dim=2, in_shape=(2, 2, 2))]
    with torch.no_grad():
        for layer in layers:
            layer.weights.zero_()
            layer.weights[:, 3] = 30
    return torch.nn.Sequential(*layers, torch.nn.Flatten(), torch.nn.Linear(8, 2))


def hybrid_config(**kwargs):
    return replace(ExperimentConfig(
        dataset="cifar10", model_variant="build_model_cifar10_compact_sixconv",
        parametrization="compact", experiment="pure_ce", iterations=20,
        temperature_warmup=0, phase_warmup=0, compact_freeze_mode="hybrid",
        compact_exploration_steps=1, compact_freeze_patience=1,
        compact_freeze_min_interval=3, compact_freeze_probe_interval=1,
        compact_freeze_probe_patience=2,
        compact_freeze_probe_batches=1, compact_freeze_loss_sigma=0,
    ), **kwargs)


def test_hybrid_spacing_confirmation_state_and_probe_is_non_mutating():
    model = toy_model()
    config = hybrid_config(compact_freeze_probe_batches=2)
    controller = HybridFreezeController(model, config)
    optimizer = torch.optim.Adam(model.parameters())
    inputs, targets = torch.rand(2, 1, 2, 2), torch.tensor([0, 1])

    def probe(step):
        return controller.probe_and_freeze(
            model, inputs, targets, step,
            additional_batches=[(inputs.flip(0), targets.flip(0))],
        )
    model(inputs).sum().backward()
    optimizer.step()
    parameters = [p.detach().clone() for p in model.parameters()]
    gradients = [p.grad.clone() for p in model.parameters()]
    moments = {p: {key: value.clone() for key, value in state.items()}
               for p, state in optimizer.state.items()}
    for step in range(1, 5):
        controller.step(step, optimizer)
    assert controller.pending_probe == "0"
    rng = torch.get_rng_state().clone()
    event = probe(4)
    assert event["passed"] and not event["frozen"]
    assert event["probe_batches"] == 2 and event["probe_examples"] == 4
    torch.testing.assert_close(torch.get_rng_state(), rng)
    for p, value, gradient in zip(model.parameters(), parameters, gradients):
        torch.testing.assert_close(p, value)
        torch.testing.assert_close(p.grad, gradient)
        for key in moments[p]:
            torch.testing.assert_close(optimizer.state[p][key], moments[p][key])
    assert all(module.training for module in model.modules())
    restored = HybridFreezeController(model, config)
    restored.load_state_dict(controller.state_dict())
    assert restored.summary() == controller.summary()
    controller = restored
    controller.step(5, optimizer)
    assert probe(5)["frozen"]
    assert model[0].is_frozen and not model[1].is_frozen
    for step in (6, 7):
        controller.step(step, optimizer)
        assert controller.pending_probe is None
    for step in (8, 9):
        controller.step(step, optimizer)
        event = probe(step)
    assert event["frozen"] and model[1].is_frozen


def test_equal_entropy_winner_and_operand_changes_are_diagnostics_not_probe_blockers():
    model = toy_model()
    layer = model[0]
    with torch.no_grad():
        layer.weights.fill_(-50)
        layer.weights[:, 3] = torch.log(torch.tensor(0.51))
        layer.weights[:, 5] = torch.log(torch.tensor(0.49))
    controller = HybridFreezeController(model, hybrid_config(
        compact_freeze_min_interval=0, compact_freeze_max_winner_change=0,
    ))
    optimizer = torch.optim.Adam(model.parameters())
    controller.step(2, optimizer)
    controller.probe_passes["0"] = 1
    entropy = layer.weights_entropy().clone()
    with torch.no_grad():
        layer.weights[:, [3, 5]] = layer.weights[:, [5, 3]]
    controller.step(3, optimizer)
    torch.testing.assert_close(layer.weights_entropy(), entropy)
    assert controller.winner_change == 1 and controller.pending_probe == "0"
    assert controller.probe_passes["0"] == 1 and controller.freeze_counters["0"] == 0
    with torch.no_grad():
        layer.impl.offsets_h[0, :, 5] += 1
    controller.step(4, optimizer)
    assert controller.winner_change == 1 and controller.pending_probe == "0"
    assert controller.probe_passes["0"] == 1
    assert controller.winner_change_window == 1


def test_noisy_entropy_does_not_block_scheduled_probes():
    model = toy_model()
    controller = HybridFreezeController(model, hybrid_config(
        compact_freeze_min_interval=0, compact_freeze_patience=200,
    ))
    optimizer = torch.optim.Adam(model.parameters())
    controller.step(2, optimizer)
    with torch.no_grad():
        model[0].weights.zero_()
    controller.step(3, optimizer)
    assert controller.entropy_delta > 5e-4
    assert controller.freeze_counters["0"] == 0
    assert controller.pending_probe == "0"


@pytest.mark.parametrize("gradient_case", ["unused", "no_grad", "zero"])
def test_missing_taylor_gradient_rejects_but_real_zero_gradient_is_valid(gradient_case):
    model = toy_model()
    layer = model[0]
    handle = None
    if gradient_case == "no_grad":
        layer.weights.requires_grad_(False)
    else:
        handle = model[1].register_forward_pre_hook(
            lambda module, inputs: (inputs[0].detach() if gradient_case == "unused" else inputs[0] * 0,)
        )
    controller = HybridFreezeController(model, hybrid_config(
        compact_freeze_min_interval=0, compact_freeze_probe_patience=1,
        compact_freeze_max_taylor=100, compact_freeze_max_loss_increase=100,
    ))
    optimizer = torch.optim.Adam(model.parameters())
    controller.step(2, optimizer)
    controller.step(3, optimizer)
    controller.probe_winners["0"] = controller.winner_signature(layer).clone()
    controller.probe_passes["0"] = 1
    try:
        event = controller.probe_and_freeze(
            model, torch.rand(2, 1, 2, 2), torch.tensor([0, 1]), 3,
        )
    finally:
        if handle is not None:
            handle.remove()
    if gradient_case == "zero":
        assert event["taylor_score"] == 0 and event["passed"] and layer.is_frozen
    else:
        assert event["taylor_score"] is None
        assert not event["passed"] and not layer.is_frozen
        assert event["consecutive_passes"] == 0


@pytest.mark.parametrize("taylor_limit,loss_limit", [(0, 10), (10, 0)])
def test_taylor_and_actual_loss_checks_can_each_reject(taylor_limit, loss_limit):
    model = toy_model()
    # Make hard identity outputs close to 1, while soft outputs are close to 0.5.
    with torch.no_grad():
        for layer in model[:2]:
            layer.weights.zero_()
            layer.weights[:, 3] = 0.001
            layer.impl.offsets_h.zero_()
            layer.impl.offsets_w.zero_()
        model[-1].weight[0].fill_(-1 / 8)
        model[-1].weight[1].fill_(1 / 8)
        model[-1].bias.zero_()
    controller = HybridFreezeController(model, hybrid_config(
        compact_freeze_min_interval=0, compact_freeze_probe_patience=1,
        compact_freeze_max_taylor=taylor_limit, compact_freeze_max_loss_increase=loss_limit,
    ))
    optimizer = torch.optim.Adam(model.parameters())
    controller.step(2, optimizer)
    controller.step(3, optimizer)
    event = controller.probe_and_freeze(model, torch.ones(2, 1, 2, 2), torch.zeros(2, dtype=torch.long), 3)
    assert event["taylor_score"] > 0 and event["loss_increase"] > 0
    assert not event["passed"] and not model[0].is_frozen


def test_sixconv_budget_shapes_cli_and_separate_mode_paths():
    args = build_parser().parse_args([
        "--dataset", "cifar10", "--model-variant", "build_model_cifar10_compact_sixconv",
        "--parametrization", "compact", "--experiment", "pure_ce", "--iterations", "900000",
        "--input-encoding", "hybrid", "--num-quantile-bits", "4", "--num-sine-bits", "1",
        "--compact-freeze-mode", "hybrid", "--compact-backend", "torch",
    ])
    config = config_from_args(args)
    config.validate()
    entropy = replace(config, compact_freeze_mode="entropy")
    entropy.validate()
    assert entropy.run_dir != config.run_dir
    assert "hybrid_freeze" not in str(entropy.run_dir)
    assert "head6_v2_plain" in str(entropy.run_dir)
    assert "hybrid_freeze_v3_probedriven_n4_sigma2" in str(config.run_dir)
    model = build_model_for_dataset(
        torch.device("cpu"), dataset="cifar10", model_variant=config.model_variant,
        parametrization="compact", input_encoding="hybrid", num_quantile_bits=4,
        num_sine_bits=1, input_thresholds=torch.tensor([[0.2, 0.4, 0.6, 0.8]]).repeat(3, 1),
        compact_backend="torch",
    )
    convs = [layer for layer in model if isinstance(layer, CandidateLogicLayer) and layer.is_convolution]
    assert [layer.out_dim for layer in convs] == [128, 128, 256, 256, 256, 256]
    assert [layer.stride for layer in convs] == [2, 1, 2, 1, 1, 1]
    assert model[1].in_shape[0] == model[0].output_channels == 15
    assert type(model) is torch.nn.Sequential
    assert not hasattr(model, "concatenation_skips")
    assert [layer.in_shape[0] for layer in convs] == [15, 128, 128, 256, 256, 256]
    assert model[8].impl.in_dim == 16384
    assert model[8].out_dim == model[9].out_dim == 80000
    report = logic_layer_width_report(model)
    assert [row["width"] for row in report] == [32768] * 2 + [16384] * 4 + [80000] * 2
    assert sum(row["lut_node_applications"] for row in report) == 291072
    assert sum(p.numel() for p in model.parameters()) == 2580480
    with torch.no_grad():
        assert model.eval()(torch.rand(1, 3, 32, 32)).shape == (1, 10)


def test_sixconv_cuda_hybrid_short_training(tmp_path, monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    pytest.importorskip("compactlogic")
    from examples.phase_logic import training

    config = hybrid_config(
        device="cuda", compact_backend="cuda", iterations=7, eval_frequency=7,
        input_encoding="hybrid", num_quantile_bits=2, num_sine_bits=1,
        compact_freeze_min_interval=0, compact_freeze_probe_patience=1,
        compact_freeze_probe_batches=2,
        compact_freeze_max_taylor=10, compact_freeze_max_loss_increase=10,
        learning_rate=1e-8, batch_size=2, output_dir=tmp_path, save_model=True,
    )
    loader = DataLoader(TensorDataset(torch.rand(2, 3, 32, 32), torch.tensor([0, 1])), batch_size=2)
    monkeypatch.setattr(training, "build_data_loaders", lambda _: (loader, loader, loader))
    monkeypatch.setattr(training, "build_cifar10_input_thresholds", lambda *_: torch.tensor(
        [[0.3, 0.6]]).repeat(3, 1))
    summary = training.run_experiment(config)
    assert summary["parameter_count"] == 2580480
    assert len(summary["logic_layer_widths"]) == 8
    assert summary["compact"]["frozen_conv_layers"] == ["1", "2", "3"]
    assert summary["compact"]["freeze_mode"] == "hybrid"
    assert (config.run_dir / "freeze_probes.csv").is_file()
    assert (config.run_dir / "best_model.pt").is_file()


@pytest.mark.parametrize("changes,sigma", [([[-0.04, -0.04], [0.03, 0.03]], 0),
                                           ([[-0.1, 0.1], [-0.1, 0.1]], 2)])
def test_multibatch_checks_reject_hidden_damage_and_uncertainty(monkeypatch, changes, sigma):
    model = toy_model()
    controller = HybridFreezeController(model, hybrid_config(
        compact_freeze_min_interval=0, compact_freeze_probe_patience=1,
        compact_freeze_probe_batches=2, compact_freeze_loss_sigma=sigma,
    ))
    optimizer = torch.optim.Adam(model.parameters())
    controller.step(2, optimizer)
    controller.step(3, optimizer)
    records = iter(changes)
    observed_weights = []

    def fake_probe(model, layer, x, y):
        observed_weights.append([p.detach().clone() for p in model.parameters()])
        delta = torch.tensor(next(records), dtype=torch.float64)
        return {"soft": 1.0, "hard": 1.0 + float(delta.mean()), "taylor": 0.0,
                "changes": delta, "increase": float(delta.mean()), "count": len(delta)}

    monkeypatch.setattr(HybridFreezeController, "_probe_batch", staticmethod(fake_probe))
    x, y = torch.rand(2, 1, 2, 2), torch.tensor([0, 1])
    with pytest.raises(ValueError, match="batches"):
        controller.probe_and_freeze(model, x, y, 3)
    event = controller.probe_and_freeze(model, x, y, 3, additional_batches=[(x, y)])
    assert event["loss_increase"] <= controller.config.compact_freeze_max_loss_increase
    assert not event["passed"] and not model[0].is_frozen
    assert (event["max_batch_loss_increase"] > 0.01 if sigma == 0
            else event["loss_upper_bound"] > 0.01)
    for before, after in zip(*observed_weights):
        torch.testing.assert_close(before, after)


def test_small_probe_drift_preserves_confirmations_but_cumulative_drift_resets(monkeypatch):
    model = toy_model()
    controller = HybridFreezeController(model, hybrid_config(
        compact_freeze_min_interval=0, compact_freeze_probe_patience=3,
        # Two toy kernels: one changed kernel is allowed, both are not.
        compact_freeze_max_probe_drift=0.5,
    ))
    optimizer = torch.optim.Adam(model.parameters())
    x, y = torch.rand(2, 1, 2, 2), torch.tensor([0, 1])

    def passing_probe(model, layer, inputs, targets):
        return {"soft": 1.0, "hard": 1.0, "taylor": 0.0,
                "changes": torch.zeros(2, dtype=torch.float64), "increase": 0.0, "count": 2}

    monkeypatch.setattr(HybridFreezeController, "_probe_batch", staticmethod(passing_probe))
    controller.step(2, optimizer)
    controller.step(3, optimizer)
    assert controller.probe_and_freeze(model, x, y, 3)["consecutive_passes"] == 1
    with torch.no_grad():
        model[0].weights[0, 5] = 40
    controller.step(4, optimizer)
    event = controller.probe_and_freeze(model, x, y, 4)
    assert event["winner_drift"] == 0.5 and not event["confirmation_reset"]
    assert event["consecutive_passes"] == 2 and not event["frozen"]
    with torch.no_grad():
        model[0].weights[1, 5] = 40
    controller.step(5, optimizer)
    event = controller.probe_and_freeze(model, x, y, 5)
    assert event["winner_drift"] == 1 and event["confirmation_reset"]
    assert event["consecutive_passes"] == 1 and not event["frozen"]
    for step in (6, 7):
        controller.step(step, optimizer)
        event = controller.probe_and_freeze(model, x, y, step)
    assert event["frozen"]
