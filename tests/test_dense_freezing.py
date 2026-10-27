"""Focused checks for opt-in dense group execution, state, and probes."""

import copy
import csv
from dataclasses import replace

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from examples.phase_logic.compact import CandidateLogicLayer
from examples.phase_logic.config import ExperimentConfig
from examples.phase_logic.dense_freezing import (
    DenseFreezeController, GroupwiseCandidateDense, enable_dense_group_freezing,
)
from examples.phase_logic.model import HybridFeatureEncoder
from torchlogix.layers import GroupSum


@pytest.fixture(autouse=True, scope="module")
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def config(**kwargs):
    return replace(ExperimentConfig(
        dataset="cifar10", model_variant="build_model_cifar10_small_compact_head",
        parametrization="compact", experiment="pure_ce", iterations=20,
        temperature_warmup=0, phase_warmup=0, compact_exploration_steps=1,
        compact_dense_freeze=True, dense_freeze_fraction=0.5,
        dense_freeze_min_interval=2, dense_freeze_probe_interval=1,
        dense_freeze_probe_patience=2, dense_freeze_probe_batches=2,
        dense_freeze_max_loss_increase=1, dense_freeze_loss_sigma=0,
    ), **kwargs)


def dense(device="cpu", backend="torch", in_dim=8, out_dim=8):
    return GroupwiseCandidateDense(CandidateLogicLayer(
        device=device, backend=backend, in_dim=in_dim, out_dim=out_dim))


@pytest.mark.parametrize("backend", ["torch", "cuda"])
def test_partial_hard_rows_do_not_drift_under_adamw(backend):
    if backend == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    device = "cuda" if backend == "cuda" else "cpu"
    layer = dense(device, backend)
    model = torch.nn.Sequential(layer)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02, weight_decay=0.3, amsgrad=True)
    controller = DenseFreezeController(model, config(), optimizer)
    x = torch.rand(3, 8, device=device, requires_grad=True)
    model(x).square().sum().backward()
    optimizer.step()
    rows = torch.tensor([0, 2], device=device)
    proposal = layer.proposal(rows)
    layer.freeze_rows(proposal, optimizer)
    frozen = layer.weights[rows].detach().clone()
    active = layer.weights[1].detach().clone()
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        x.grad = None
        output = model.train()(x)
        torch.testing.assert_close(output[:, rows], layer.hard_rows(x, proposal["pairs"], proposal["codes"]))
        output.square().sum().backward()
        assert torch.count_nonzero(layer.weights.grad[rows]) == 0
        assert torch.count_nonzero(x.grad) > 0
        optimizer.step()
        assert torch.equal(layer.weights[rows], frozen)
        for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            assert torch.count_nonzero(optimizer.state[layer.weights][key][rows]) == 0
    assert not torch.equal(layer.weights[1], active)
    assert layer.resample(rows, optimizer) == (0, 0)
    assert not controller.fully_frozen


def test_probes_are_paired_nonmutating_and_pin_winners_across_confirmations():
    class Counter(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("calls", torch.zeros(()))

        def forward(self, x):
            self.calls.add_(1)
            return x

    model = torch.nn.Sequential(Counter(), torch.nn.Dropout(0.3), dense(), GroupSum(k=2, tau=1))
    cfg = config(dense_freeze_min_interval=0, dense_freeze_max_loss_increase=100)
    optimizer = torch.optim.Adam(model.parameters())
    controller = DenseFreezeController(model, cfg, optimizer)
    batches = [(torch.rand(3, 8), torch.tensor([0, 1, 0])),
               (torch.rand(3, 8), torch.tensor([1, 0, 1]))]
    controller.step(2)
    pinned = copy.deepcopy(controller.proposal)
    rng = torch.get_rng_state().clone()
    state = copy.deepcopy(model.state_dict())
    event = controller.probe_and_freeze(model, batches, 2, optimizer)
    assert event["passed"] and not event["committed"]
    assert torch.equal(torch.get_rng_state(), rng)
    for key, value in model.state_dict().items():
        assert torch.equal(value, state[key])
    assert all(module.training for module in model.modules())
    saved = controller.state_dict()
    restored = DenseFreezeController(model, cfg, torch.optim.Adam(model.parameters()))
    restored.load_state_dict(saved)
    torch.testing.assert_close(restored.proposal["pairs"], pinned["pairs"])
    # Training can move the current argmax; the next probe must still evaluate
    # and commit the SAME pinned choice, not substitute an untested new winner.
    with torch.no_grad():
        model[2].weights[pinned["rows"], 0] = 40
    controller.step(3)
    torch.testing.assert_close(controller.proposal["logits"], pinned["logits"])
    event = controller.probe_and_freeze(model, batches, 3, optimizer)
    assert event["committed"] and model[2].num_frozen_rows == 4
    torch.testing.assert_close(model[2].frozen_codes[pinned["rows"]], pinned["codes"])
    model[1].eval()
    x = batches[0][0]
    torch.testing.assert_close(model[2](x)[:, pinned["rows"]],
                               model[2].hard_rows(x, pinned["pairs"], pinned["codes"]))


def test_recovery_layer_order_and_frozen_state_roundtrip():
    def make():
        model = torch.nn.Sequential(
            CandidateLogicLayer(device="cpu", backend="torch", in_dim=4, out_dim=2, in_shape=(1, 2, 2)),
            torch.nn.Flatten(), dense(), dense(in_dim=8, out_dim=4), GroupSum(k=2, tau=1))
        with torch.no_grad():
            for layer in (model[0], model[2], model[3]):
                layer.weights.zero_()
                layer.weights[:, 3] = 30
        return model

    model = make()
    cfg = config()
    optimizer = torch.optim.Adam(model.parameters())
    controller = DenseFreezeController(model, cfg, optimizer)
    batches = [(torch.rand(2, 1, 2, 2), torch.tensor([0, 1])) for _ in range(2)]
    controller.step(2)
    assert not controller.pending_probe
    model[0].freeze()
    commits = []
    for step in range(3, 18):
        controller.step(step)
        if controller.pending_probe:
            event = controller.probe_and_freeze(model, batches, step, optimizer)
            if event["committed"]:
                commits.append((step, event["layer"]))
                if event["layer"] == "2":
                    assert model[3].num_frozen_rows == 0
    assert [name for _, name in commits] == ["2", "2", "3", "3"]
    assert all(b - a >= cfg.dense_freeze_min_interval for (a, _), (b, _) in zip(commits, commits[1:]))
    assert controller.fully_frozen
    assert not model.train()(batches[0][0]).requires_grad
    restored = make()
    restored.load_state_dict(model.state_dict())
    restored_optimizer = torch.optim.Adam(restored.parameters())
    restored_controller = DenseFreezeController(restored, cfg, restored_optimizer)
    restored_controller.load_state_dict(controller.state_dict())
    assert restored_controller.summary() == controller.summary()
    torch.testing.assert_close(restored.train()(batches[0][0]), model(batches[0][0]), atol=0, rtol=0)


@pytest.mark.parametrize("delta", [1.0, float("nan")])
def test_rejected_groups_do_not_freeze(delta, monkeypatch):
    model = torch.nn.Sequential(dense(), GroupSum(k=2, tau=1))
    cfg = config(dense_freeze_min_interval=0, dense_freeze_max_loss_increase=0.01)
    optimizer = torch.optim.Adam(model.parameters())
    controller = DenseFreezeController(model, cfg, optimizer)
    controller.step(2)
    def bad_probe(*args):
        return {"soft": 1.0, "hard": 1 + delta, "changes": torch.tensor([delta, delta]), "count": 2}
    monkeypatch.setattr(controller, "_probe_batch", bad_probe)
    event = controller.probe_and_freeze(model, [(torch.rand(2, 8), torch.tensor([0, 1]))] * 2, 2, optimizer)
    assert not event["passed"] and not event["committed"]
    assert model[0].num_frozen_rows == 0 and controller.proposal is None


def test_training_integration_and_cli_validation(tmp_path, monkeypatch):
    import examples.phase_logic.training as training
    from examples.phase_logic.run_experiment import build_parser, config_from_args
    args = build_parser().parse_args([
        "--dataset", "cifar10", "--model-variant", "build_model_cifar10_small_compact_head",
        "--parametrization", "compact", "--experiment", "pure_ce", "--iterations", "700000",
        "--compact-dense-freeze",
    ])
    parsed = config_from_args(args)
    parsed.validate()
    assert "dense_groups_v1" in str(parsed.run_dir)
    with pytest.raises(ValueError, match="at least two"):
        replace(parsed, dense_freeze_probe_batches=1).validate()
    thresholds = torch.tensor([[0.3, 0.7]]).expand(3, -1)
    model = torch.nn.Sequential(
        HybridFeatureEncoder(mode="hybrid_lbp", thresholds=thresholds),
        CandidateLogicLayer(device="cpu", backend="torch", in_dim=48, out_dim=2, in_shape=(12, 2, 2)),
        torch.nn.Flatten(), dense(), dense(in_dim=8, out_dim=4), GroupSum(k=2, tau=1))
    with torch.no_grad():
        for layer in (model[1], model[3], model[4]):
            layer.weights.zero_()
            layer.weights[:, 3] = 30
    dataset = TensorDataset(torch.rand(4, 3, 2, 2), torch.tensor([0, 1, 0, 1]))
    loader = DataLoader(dataset, batch_size=2)
    monkeypatch.setattr(training, "build_data_loaders", lambda config: (loader, loader, loader))
    monkeypatch.setattr(training, "build_cifar10_input_thresholds", lambda *args: thresholds)
    monkeypatch.setattr(training, "build_model_for_dataset", lambda *args, **kwargs: model)
    cfg = config(output_dir=tmp_path, iterations=12, device="cpu", batch_size=2, eval_frequency=2,
                 input_encoding="hybrid", hybrid_features="lbp", compact_freeze_patience=1,
                 dense_freeze_min_interval=0, dense_freeze_probe_patience=1,
                 dense_freeze_max_loss_increase=100, save_model=True)
    summary = training.run_experiment(cfg)
    assert summary["dense_freezing"]["next_layer"] is None
    with (cfg.run_dir / "dense_freeze_probes.csv").open() as handle:
        events = list(csv.DictReader(handle))
    assert len(events) == 4 and all(event["committed"] == "True" for event in events)
    checkpoint = torch.load(cfg.run_dir / "best_model.pt", weights_only=True)
    assert "dense_freezer_state" in checkpoint and "optimizer_state" in checkpoint
    final = torch.load(cfg.run_dir / "final_model.pt", weights_only=True)
    assert final["3.frozen_rows"].all() and final["4.frozen_rows"].all()
