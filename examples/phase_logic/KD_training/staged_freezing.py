"""Mapping-stability staged freezing for LUT6 student convolution layers."""

from __future__ import annotations

from pathlib import Path

import torch


STAGED_FREEZE_FIELDS = (
    "epoch", "target", "action", "mapping_change", "mapping_changes",
    "allowed_mapping_changes", "mapping_size", "stable_count", "observe_count",
    "monitor_count", "baseline_accuracy", "baseline_loss",
    "monitor_accuracy", "monitor_loss", "reason", "safe_checkpoint",
)

def _mapping_phases(model):
    """Discover the sequential convolution mappings exposed by a student."""
    phases = []
    index = 1
    while hasattr(model, f"conv{index}"):
        path = f"conv{index}.mapping_logits"
        _resolve_parameter(model, path)
        phases.append((f"conv{index}_mapping", path))
        index += 1
    if not phases:
        raise ValueError("Staged freezing requires at least one convolution mapping.")
    return tuple(phases)


def _resolve_parameter(model, path: str) -> torch.nn.Parameter:
    value = model
    for part in path.split("."):
        value = getattr(value, part)
    if not isinstance(value, torch.nn.Parameter):
        raise TypeError(f"{path} is not a trainable parameter.")
    return value


class DiscreteStabilityStagedFreezer:
    """Freeze stable connection mappings one layer at a time with rollback.

    LUT truth tables remain trainable. Freezing a mapping does not alter the
    hard forward pass, so validation is monitored for several subsequent epochs
    to measure the opportunity cost of preventing further mapping updates.
    """

    def __init__(
        self,
        model,
        optimizer,
        safe_checkpoint,
        *,
        stability_patience: int = 10,
        max_mapping_change: float = 0.001,
        monitor_epochs: int = 5,
        max_accuracy_drop: float = 0.015,
        max_loss_increase: float = 0.05,
        retry_cooldown: int = 10,
        max_observe_epochs: int = 100,
    ):
        if (stability_patience <= 0 or monitor_epochs <= 0 or retry_cooldown < 0
                or max_observe_epochs <= 0):
            raise ValueError("Stability/monitor durations must be positive and cooldown nonnegative.")
        for name, value in (
            ("max_mapping_change", max_mapping_change),
            ("max_accuracy_drop", max_accuracy_drop),
            ("max_loss_increase", max_loss_increase),
        ):
            if value < 0:
                raise ValueError(f"{name} must be nonnegative.")
        self.model = model
        self.optimizer = optimizer
        self.safe_checkpoint = Path(safe_checkpoint)
        self.stability_patience = stability_patience
        self.max_mapping_change = max_mapping_change
        self.monitor_epochs = monitor_epochs
        self.max_accuracy_drop = max_accuracy_drop
        self.max_loss_increase = max_loss_increase
        self.retry_cooldown = retry_cooldown
        self.max_observe_epochs = max_observe_epochs
        self.mapping_phases = _mapping_phases(model)
        self.phase_index = 0
        self.stable_count = 0
        self.observe_count = 0
        self.previous_signature = None
        self.monitoring = False
        self.baseline_accuracy = None
        self.baseline_loss = None
        self.monitor_metrics = []
        self.retry_after_epoch = 0
        self.events = []

    @property
    def complete(self) -> bool:
        return self.phase_index >= len(self.mapping_phases)

    @property
    def target(self) -> str | None:
        return None if self.complete else self.mapping_phases[self.phase_index][0]

    def _current_parameter(self) -> torch.nn.Parameter:
        return _resolve_parameter(self.model, self.mapping_phases[self.phase_index][1])

    @torch.no_grad()
    def _signature(self) -> torch.Tensor:
        return self._current_parameter().argmax(-1).detach().clone()

    def _event(self, *, epoch, action, change=None, changes=None, allowed=None,
               mapping_size=None, reason="", allow_best=True, target=None):
        monitor_accuracy = (
            sum(item[0] for item in self.monitor_metrics) / len(self.monitor_metrics)
            if self.monitor_metrics else None
        )
        monitor_loss = (
            sum(item[1] for item in self.monitor_metrics) / len(self.monitor_metrics)
            if self.monitor_metrics else None
        )
        event = {
            "epoch": epoch,
            "target": target or self.target or "complete",
            "action": action,
            "mapping_change": change,
            "mapping_changes": changes,
            "allowed_mapping_changes": allowed,
            "mapping_size": mapping_size,
            "stable_count": self.stable_count,
            "observe_count": self.observe_count,
            "monitor_count": len(self.monitor_metrics),
            "baseline_accuracy": self.baseline_accuracy,
            "baseline_loss": self.baseline_loss,
            "monitor_accuracy": monitor_accuracy,
            "monitor_loss": monitor_loss,
            "reason": reason,
            "safe_checkpoint": str(self.safe_checkpoint),
            "allow_best_checkpoint": allow_best,
        }
        self.events.append(dict(event))
        return event

    def _save_safe(self, epoch, metrics):
        self.safe_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "checkpoint_kind": "student_mapping_freeze_safe_v1",
            "epoch": int(epoch),
            "student_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "phase_index": self.phase_index,
            "validation": dict(metrics),
        }, self.safe_checkpoint)

    def _apply_freeze_state(self):
        for index, (_, path) in enumerate(self.mapping_phases):
            if index < self.phase_index:
                _resolve_parameter(self.model, path).requires_grad_(False)
            else:
                _resolve_parameter(self.model, path).requires_grad_(True)

    def _restore_safe(self):
        checkpoint = torch.load(self.safe_checkpoint, map_location="cpu", weights_only=True)
        if checkpoint.get("checkpoint_kind") != "student_mapping_freeze_safe_v1":
            raise ValueError("Invalid staged-freezing safe checkpoint.")
        self.model.load_state_dict(checkpoint["student_state"], strict=True)
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self._apply_freeze_state()

    def observe(self, *, epoch: int, accuracy: float, loss: float) -> dict:
        """Observe one validation result and possibly freeze/accept/rollback."""
        metrics = {"accuracy": float(accuracy), "loss": float(loss)}
        if self.complete:
            return self._event(
                epoch=epoch, action="complete",
                reason="all configured convolution mappings frozen",
            )

        if self.monitoring:
            self.monitor_metrics.append((metrics["accuracy"], metrics["loss"]))
            if len(self.monitor_metrics) < self.monitor_epochs:
                return self._event(
                    epoch=epoch, action="monitor", reason="collecting post-freeze validation",
                    allow_best=False,
                )
            mean_accuracy = sum(item[0] for item in self.monitor_metrics) / len(self.monitor_metrics)
            mean_loss = sum(item[1] for item in self.monitor_metrics) / len(self.monitor_metrics)
            failed = (
                mean_accuracy < self.baseline_accuracy - self.max_accuracy_drop
                or mean_loss > self.baseline_loss + self.max_loss_increase
            )
            old_target = self.target
            if failed:
                self._restore_safe()
                self.monitoring = False
                self.stable_count = 0
                self.observe_count = 0
                self.previous_signature = self._signature()
                self.retry_after_epoch = epoch + self.retry_cooldown
                event = self._event(
                    epoch=epoch, action="rollback",
                    reason=f"{old_target} post-freeze validation regressed",
                    allow_best=False, target=old_target,
                )
                self.baseline_accuracy = self.baseline_loss = None
                self.monitor_metrics = []
                return event
            self.phase_index += 1
            self.monitoring = False
            self.stable_count = 0
            self.observe_count = 0
            self.previous_signature = None
            event = self._event(
                epoch=epoch, action="accept",
                reason=f"{old_target} freeze passed validation monitoring",
                target=old_target,
            )
            self.baseline_accuracy = self.baseline_loss = None
            self.monitor_metrics = []
            return event

        signature = self._signature()
        in_cooldown = epoch < self.retry_after_epoch
        if not in_cooldown:
            self.observe_count += 1
        change = None
        changes = None
        mapping_size = signature.numel()
        allowed = (
            max(1, int(self.max_mapping_change * mapping_size))
            if self.max_mapping_change > 0 else 0
        )
        if self.previous_signature is not None:
            changes = int((signature != self.previous_signature).sum())
            change = changes / mapping_size
            self.stable_count = self.stable_count + 1 if changes <= allowed else 0
        self.previous_signature = signature
        if in_cooldown:
            return self._event(epoch=epoch, action="cooldown", change=change, changes=changes,
                               allowed=allowed, mapping_size=mapping_size,
                               reason="waiting after rollback")
        stable = self.stable_count >= self.stability_patience
        timed_out = self.observe_count >= self.max_observe_epochs
        if not stable and not timed_out:
            return self._event(epoch=epoch, action="observe", change=change, changes=changes,
                               allowed=allowed, mapping_size=mapping_size,
                               reason="mapping has not met the stability patience")

        self._save_safe(epoch, metrics)
        self._current_parameter().requires_grad_(False)
        self.monitoring = True
        self.baseline_accuracy = metrics["accuracy"]
        self.baseline_loss = metrics["loss"]
        self.monitor_metrics = []
        return self._event(epoch=epoch, action="freeze_start", change=change, changes=changes,
                           allowed=allowed, mapping_size=mapping_size,
                           reason=("mapping stability threshold reached" if stable else
                                   "maximum observation window reached; validation-protected freeze attempt"))

    def summary(self) -> dict:
        return {
            "mode": "mapping_discrete_stability_v2",
            "frozen_mappings": [name for name, _ in self.mapping_phases[:self.phase_index]],
            "current_target": self.target,
            "monitoring": self.monitoring,
            "stable_count": self.stable_count,
            "observe_count": self.observe_count,
            "stability_patience": self.stability_patience,
            "max_mapping_change": self.max_mapping_change,
            "small_layer_minimum_allowed_changes": 1 if self.max_mapping_change > 0 else 0,
            "monitor_epochs": self.monitor_epochs,
            "max_accuracy_drop": self.max_accuracy_drop,
            "max_loss_increase": self.max_loss_increase,
            "retry_cooldown": self.retry_cooldown,
            "max_observe_epochs": self.max_observe_epochs,
        }
