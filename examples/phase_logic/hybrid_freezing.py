"""Opt-in, loss-sensitive freezing; the original CompactLogic controller is unchanged."""

import math
from collections import deque

import torch
from torch.nn import functional as F

from .compact import CompactLogicController


FREEZE_PROBE_FIELDS = (
    "step", "layer", "entropy_delta", "entropy_drift_ema", "winner_change",
    "winner_change_window", "winner_drift", "confirmation_reset", "taylor_score",
    "loss_soft", "loss_hard", "loss_increase", "probe_batches", "probe_examples",
    "max_batch_taylor", "max_batch_loss_increase", "loss_stderr", "loss_upper_bound",
    "passed", "consecutive_passes", "frozen",
)


class HybridFreezeController(CompactLogicController):
    """Probe-driven shallow-to-deep freezing, with stability diagnostics.

    Probes estimate the loss effect of hardening the *next whole layer*, not
    pruning it. They do not accumulate parameter gradients or update Adam state.
    No test/validation examples participate in freeze decisions.
    """

    def __init__(self, model, config):
        super().__init__(model, config)
        self.config = config
        self.previous_winners = {}
        self.probe_passes = {}
        self.probe_winners = {}
        self.last_freeze_step = self.exploration_steps
        self.pending_probe = None
        self.last_probe = None
        self.entropy_delta = None
        self.winner_change = None
        self.winner_change_window = None
        self.entropy_drift_ema = None
        self.winner_windows = {}
        self.entropy_drift_emas = {}

    @staticmethod
    @torch.no_grad()
    def winner_signature(layer):
        """Actual opcode and two spatial input coordinates, not candidate-slot IDs."""
        impl = layer.impl
        winner = layer.weights.argmax(-1)
        rows = torch.arange(winner.numel(), device=winner.device)
        parts = [impl.gate_sequence[rows, winner, None].to(torch.int64)]
        for offsets in (impl.offsets_ch, impl.offsets_h, impl.offsets_w):
            parts.append(offsets[:, rows, winner].T.to(torch.int64))
        return torch.cat(parts, dim=1)

    @torch.no_grad()
    def step(self, step, optimizer):
        self.pending_probe = None
        if step <= self.exploration_steps:
            return super().step(step, optimizer)
        for name, layer in self.convs:
            if layer.is_frozen:
                continue
            entropy = float(layer.weights_entropy().mean())
            signature = self.winner_signature(layer)
            if name not in self.freeze_ema:
                self.freeze_ema[name] = entropy
                self.freeze_counters[name] = 0
                self.previous_winners[name] = signature
                self.entropy_delta = self.winner_change = None
                self.winner_change_window = self.entropy_drift_ema = None
                return
            old = self.freeze_ema[name]
            self.entropy_delta = abs(entropy - old)
            self.winner_change = float(
                (signature != self.previous_winners[name]).any(dim=1).float().mean()
            )
            self.previous_winners[name] = signature
            self.freeze_ema[name] = 0.99 * old + 0.01 * entropy
            window = self.winner_windows.setdefault(name, deque(maxlen=self.freeze_patience))
            window.append(self.winner_change)
            self.winner_change_window = sum(window) / len(window)
            self.entropy_drift_ema = (
                0.99 * self.entropy_drift_emas.get(name, self.entropy_delta)
                + 0.01 * self.entropy_delta
            )
            self.entropy_drift_emas[name] = self.entropy_drift_ema
            # Retain the old stability counter for diagnostics only. Small
            # fluctuations must not prevent probes or erase passed loss checks.
            stable = (self.entropy_delta < 5e-4
                      and self.winner_change <= self.config.compact_freeze_max_winner_change)
            self.freeze_counters[name] = self.freeze_counters[name] + 1 if stable else 0
            if (step - self.last_freeze_step >= self.config.compact_freeze_min_interval
                    and step % self.config.compact_freeze_probe_interval == 0):
                self.pending_probe = name
            return

    @staticmethod
    def _probe_batch(model, layer, inputs, targets):
        """Evaluate the same whole-layer intervention on one paired batch."""
        captured = {}

        def capture(module, args, output):
            captured["soft"] = output
            was_training = module.impl.training
            try:
                module.impl.eval()
                with torch.no_grad():
                    captured["hard"] = module.impl(args[0])
            finally:
                module.impl.train(was_training)

        handle = layer.register_forward_hook(capture)
        try:
            losses_soft = F.cross_entropy(model(inputs), targets, reduction="none")
        finally:
            handle.remove()
        gradient = None
        if losses_soft.requires_grad and captured["soft"].requires_grad:
            gradient, = torch.autograd.grad(losses_soft.mean(), captured["soft"], allow_unused=True)
        if gradient is None:
            # Missing sensitivity is not evidence of safety. The finite-score
            # guard below rejects the probe and resets freeze confirmations.
            taylor = math.inf
        else:
            delta = captured["hard"] - captured["soft"].detach()
            # CE uses a batch mean: summing per-example absolute inner
            # products gives mean |g_example dot (hard-soft)|.
            taylor = float((gradient * delta).flatten(1).sum(1).abs().sum())
        handle = layer.register_forward_hook(lambda module, args, output: captured["hard"])
        try:
            with torch.no_grad():
                losses_hard = F.cross_entropy(model(inputs), targets, reduction="none")
        finally:
            handle.remove()
        changes = (losses_hard - losses_soft.detach()).detach().cpu().double()
        return {
            "soft": float(losses_soft.detach().mean()), "hard": float(losses_hard.mean()),
            "taylor": taylor, "changes": changes, "increase": float(changes.mean()),
            "count": targets.numel(),
        }

    def probe_and_freeze(self, model, inputs, targets, step, *, additional_batches=()):
        """Check multiple batches at one fixed snapshot, without optimizer updates."""
        name = self.pending_probe
        if name is None:
            return None
        batches = ((inputs, targets), *additional_batches)
        if len(batches) != self.config.compact_freeze_probe_batches:
            raise ValueError("The hybrid probe must receive compact_freeze_probe_batches batches.")
        if any(y.numel() == 0 or x.device != inputs.device for x, y in batches):
            raise ValueError("Hybrid probe batches must be nonempty and use the same device.")
        self.pending_probe = None
        layer = dict(self.convs)[name]
        signature = self.winner_signature(layer)
        previous = self.probe_winners.get(name)
        winner_drift = (float((signature != previous).any(dim=1).float().mean())
                        if previous is not None else 0.0)
        confirmation_reset = (previous is not None
                              and winner_drift > self.config.compact_freeze_max_probe_drift)
        if previous is None or self.probe_passes.get(name, 0) == 0 or confirmation_reset:
            # Anchor a confirmation streak to its first candidate assignment.
            # Allow small drift, but prevent cumulative changes from creeping
            # past the tolerance through a sequence of small per-probe changes.
            self.probe_passes[name] = 0
            self.probe_winners[name] = signature.clone()
        modes = [(module, module.training) for module in model.modules()]
        devices = [inputs.device.index] if inputs.is_cuda else []
        try:
            # Graphs are released batch-by-batch, avoiding a large combined batch.
            with torch.random.fork_rng(devices=devices), torch.enable_grad():
                model.train()
                results = [self._probe_batch(model, layer, x, y) for x, y in batches]
        finally:
            for module, training in modes:
                module.training = training

        count = sum(result["count"] for result in results)
        changes = torch.cat([result["changes"] for result in results])
        soft, hard, taylor = (
            sum(result[key] * result["count"] for result in results) / count
            for key in ("soft", "hard", "taylor")
        )
        increase = float(changes.mean())
        stderr = float(changes.std(unbiased=True) / math.sqrt(count)) if count > 1 else math.inf
        upper = (increase + self.config.compact_freeze_loss_sigma * stderr
                 if self.config.compact_freeze_loss_sigma > 0 else increase)
        max_taylor = max(result["taylor"] for result in results)
        max_increase = max(result["increase"] for result in results)
        finite = all(math.isfinite(result[key]) for result in results
                     for key in ("soft", "hard", "taylor", "increase")) and math.isfinite(upper)
        passed = (finite and max_taylor <= self.config.compact_freeze_max_taylor
                  and max_increase <= self.config.compact_freeze_max_loss_increase
                  and upper <= self.config.compact_freeze_max_loss_increase)
        self.probe_passes[name] = self.probe_passes.get(name, 0) + 1 if passed else 0
        frozen = self.probe_passes[name] >= self.config.compact_freeze_probe_patience
        if frozen:
            layer.freeze()
            self.last_freeze_step = step
            print(f"  compact hybrid freeze step={step} layer={name}")
        self.last_probe = {
            "step": step, "layer": name, "entropy_delta": self.entropy_delta,
            "entropy_drift_ema": self.entropy_drift_ema,
            "winner_change": self.winner_change,
            "winner_change_window": self.winner_change_window,
            "winner_drift": winner_drift, "confirmation_reset": confirmation_reset,
            "taylor_score": taylor if math.isfinite(taylor) else None,
            "loss_soft": soft if math.isfinite(soft) else None,
            "loss_hard": hard if math.isfinite(hard) else None,
            "loss_increase": increase if math.isfinite(increase) else None,
            "probe_batches": len(batches), "probe_examples": count,
            "max_batch_taylor": max_taylor if math.isfinite(max_taylor) else None,
            "max_batch_loss_increase": max_increase if math.isfinite(max_increase) else None,
            "loss_stderr": stderr if math.isfinite(stderr) else None,
            "loss_upper_bound": upper if math.isfinite(upper) else None,
            "passed": passed, "consecutive_passes": self.probe_passes[name], "frozen": frozen,
        }
        return dict(self.last_probe)

    def summary(self):
        return {
            **super().summary(), "freeze_mode": "hybrid",
            "next_freeze_layer": next((name for name, layer in self.convs if not layer.is_frozen), None),
            "entropy_delta": self.entropy_delta, "winner_change": self.winner_change,
            "entropy_drift_ema": self.entropy_drift_ema,
            "winner_change_window": self.winner_change_window,
            "freeze_eligibility": "probe-driven",
            "stable_steps": max((self.freeze_counters.get(name, 0) for name, layer in self.convs
                                 if not layer.is_frozen), default=0),
            "last_freeze_probe": self.last_probe,
        }

    def state_dict(self):
        return {
            **super().state_dict(),
            "hybrid_state": {
                "previous_winners": {name: value.clone() for name, value in self.previous_winners.items()},
                "probe_winners": {name: value.clone() for name, value in self.probe_winners.items()},
                "probe_passes": dict(self.probe_passes), "last_freeze_step": self.last_freeze_step,
                "last_probe": self.last_probe, "entropy_delta": self.entropy_delta,
                "winner_change": self.winner_change,
                "winner_change_window": self.winner_change_window,
                "entropy_drift_ema": self.entropy_drift_ema,
                "winner_windows": {name: list(window) for name, window in self.winner_windows.items()},
                "entropy_drift_emas": dict(self.entropy_drift_emas),
            },
        }

    def load_state_dict(self, state):
        super().load_state_dict(state)
        hybrid = state["hybrid_state"]
        devices = {name: layer.weights.device for name, layer in self.convs}
        self.previous_winners = {name: value.clone().to(devices[name])
                                 for name, value in hybrid["previous_winners"].items()}
        self.probe_passes = dict(hybrid["probe_passes"])
        self.probe_winners = {name: value.clone().to(devices[name])
                              for name, value in hybrid.get("probe_winners", {}).items()}
        self.last_freeze_step = hybrid["last_freeze_step"]
        self.last_probe = hybrid["last_probe"]
        self.entropy_delta = hybrid["entropy_delta"]
        self.winner_change = hybrid["winner_change"]
        self.winner_change_window = hybrid.get("winner_change_window")
        self.entropy_drift_ema = hybrid.get("entropy_drift_ema")
        self.winner_windows = {name: deque(values, maxlen=self.freeze_patience)
                               for name, values in hybrid.get("winner_windows", {}).items()}
        self.entropy_drift_emas = dict(hybrid.get("entropy_drift_emas", {}))
        self.pending_probe = None
