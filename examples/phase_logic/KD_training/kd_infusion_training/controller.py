"""Validation-protected LUT6 proposals; independent of the KD training loop."""

from dataclasses import asdict
import math
from pathlib import Path

import torch

# Reuse only the immutable mixtures. CompactLogic's candidate resampler is
# incompatible with the student's one learned 64-entry table per node.
from ...infusion_training.controller import INFUSION_LEVELS


INFUSION_LOG_FIELDS = (
    "epoch", "step", "action", "from_level", "to_level", "accuracy", "loss",
    "safe_accuracy", "safe_loss", "layer", "rows", "uniform", "identity",
    "inversion", "constant", "reason",
)


def dense_lut6_tables(model):
    """Discover root or nested dense tables, including dense2_tables aliases."""
    found, seen = {}, set()
    for module_name, module in model.named_modules():
        if hasattr(module, "in_channels") and hasattr(module, "out_channels"):
            continue  # Preserve convolutional features; no LUT2 prior is assumed.
        for name, parameter in module.named_parameters(recurse=False):
            if (name == "tables" or name.endswith("_tables")) and id(parameter) not in seen:
                if parameter.ndim != 2 or parameter.shape[1] != 64:
                    raise ValueError(f"{module_name}.{name}: infusion supports LUT6 tables only.")
                found[f"{module_name}.{name}".lstrip(".")] = parameter
                seen.add(id(parameter))
    if not found:
        raise ValueError("KD infusion requires at least one dense LUT6 table parameter.")
    return found


class KDProgressiveAggressiveInfusion:
    """Propose simple dense functions, escalating only near best validation accuracy.

    This replaces a bounded number of existing weak tables, not input mappings
    or a percentage of all gates. The proposals remain trainable afterward.
    """

    def __init__(self, model, optimizer, safe_checkpoint, *, warmup_epochs=100,
                 interval_epochs=10, fraction=0.01, minimum_transition_steps=15_000):
        if warmup_epochs < 1 or interval_epochs < 1 or minimum_transition_steps < 0:
            raise ValueError("Invalid infusion warmup, interval, or cooldown.")
        if not math.isfinite(fraction) or not 0 < fraction <= 1:
            raise ValueError("Infusion fraction must be in (0,1].")
        self.model, self.optimizer = model, optimizer
        self.safe_checkpoint = Path(safe_checkpoint)
        self.tables = dense_lut6_tables(model)
        self.options = dict(warmup_epochs=warmup_epochs, interval_epochs=interval_epochs,
                            fraction=fraction, minimum_transition_steps=minimum_transition_steps)
        self.level = 1
        self.stable_count = 0
        self.last_accuracy = self.last_loss = self.best_accuracy = None
        self.safe_accuracy = self.safe_loss = self.best_safe_accuracy = None
        self.last_transition_step = None
        self.retry_after_step = 0
        self.total_proposed_rows = 0

    @property
    def policy(self):
        return INFUSION_LEVELS[self.level - 1]

    def _event(self, epoch, step, action, accuracy=None, loss=None, **extra):
        return dict(epoch=epoch, step=step, action=action, from_level=self.level,
                    to_level=self.level, accuracy=accuracy, loss=loss,
                    safe_accuracy=self.safe_accuracy, safe_loss=self.safe_loss, **extra)

    def _save_safe(self, accuracy, loss):
        from .checkpoint import save_safe_checkpoint

        self.safe_accuracy, self.safe_loss = accuracy, loss
        self.best_safe_accuracy = max(accuracy, self.best_safe_accuracy or accuracy)
        save_safe_checkpoint(self)

    def observe(self, *, epoch, step, accuracy, loss):
        if not math.isfinite(accuracy) or not math.isfinite(loss) or not 0 <= accuracy <= 1:
            raise ValueError("Infusion needs finite discrete validation accuracy and CE loss.")
        if epoch < self.options["warmup_epochs"]:
            self.best_accuracy = max(accuracy, self.best_accuracy or accuracy)
            return self._event(epoch, step, "warmup", accuracy, loss)
        if self.safe_accuracy is not None and (
            accuracy < self.safe_accuracy - 0.015 or loss > self.safe_loss + 0.10
        ):
            from .checkpoint import restore_safe_checkpoint

            old_level, proposed_rows = self.level, self.total_proposed_rows
            restore_safe_checkpoint(self)
            self.total_proposed_rows = proposed_rows
            self.stable_count = 0
            self.last_transition_step = step
            self.retry_after_step = step + self.options["minimum_transition_steps"]
            event = self._event(epoch, step, "rollback", accuracy, loss,
                                reason="validation regressed beyond the safe checkpoint")
            event["from_level"] = old_level
            return event
        self.best_accuracy = max(accuracy, self.best_accuracy or accuracy)
        stable = (self.last_accuracy is None or (
            accuracy >= self.last_accuracy - 0.003 and loss <= self.last_loss + 0.05
        ))
        reference = max(self.best_accuracy, self.best_safe_accuracy or self.best_accuracy)
        self.stable_count = self.stable_count + 1 if stable and accuracy >= reference - 0.003 else 0
        self.last_accuracy, self.last_loss = accuracy, loss
        if self.safe_accuracy is None:
            self._save_safe(accuracy, loss)  # Protect Level 1 before the first proposal.
        cooldown = (step >= self.retry_after_step and (
            self.last_transition_step is None
            or step - self.last_transition_step >= self.options["minimum_transition_steps"]
        ))
        if self.level < len(INFUSION_LEVELS) and self.stable_count >= 3 and cooldown:
            self._save_safe(accuracy, loss)  # Persist old level before changing it.
            event = self._event(epoch, step, "advance", accuracy, loss,
                                reason="three stable evaluations near best safe accuracy")
            self.level += 1
            self.stable_count = 0
            self.last_transition_step = step
            event["to_level"] = self.level
            return event
        return self._event(epoch, step, "hold", accuracy, loss)

    @torch.no_grad()
    def infuse(self, *, epoch, step):
        if (epoch < self.options["warmup_epochs"] or step < self.retry_after_step
                or (epoch - self.options["warmup_epochs"]) % self.options["interval_epochs"]):
            return []
        if self.safe_accuracy is None:
            raise RuntimeError("Observe validation before proposing any truth-table changes.")
        events = []
        for name, table in self.tables.items():
            if not table.requires_grad:
                continue
            bits = table >= 0
            addresses = torch.arange(64, device=table.device)
            pins = ((addresses[None, :] >> torch.arange(6, device=table.device)[:, None]) & 1).bool()
            constant = bits.all(-1) | (~bits).all(-1)
            identity = ((bits[:, None, :] == pins).all(-1)
                        | (bits[:, None, :] != pins).all(-1)).any(-1)
            eligible = (~(constant | identity)).nonzero(as_tuple=True)[0]
            count = min(len(eligible), int(table.shape[0] * self.options["fraction"]))
            if count == 0:
                continue
            confidence = table[eligible].abs().mean(-1)
            rows = eligible[confidence.topk(count, largest=False).indices]
            p = self.policy
            categories = torch.multinomial(table.new_tensor(
                [p.uniform, p.identity, p.inversion, p.constant]), count, replacement=True)
            proposals = torch.rand((count, 64), device=table.device) >= 0.5
            selected_pins = pins[torch.randint(6, (count,), device=table.device)]
            proposals[categories == 1] = selected_pins[categories == 1]
            proposals[categories == 2] = ~selected_pins[categories == 2]
            constants = torch.randint(2, (count, 1), device=table.device).bool().expand(-1, 64)
            proposals[categories == 3] = constants[categories == 3]
            table[rows] = proposals.to(table.dtype) - 0.5
            # Adam momentum from the discarded functions must not drive proposals.
            for value in self.optimizer.state.get(table, {}).values():
                if torch.is_tensor(value) and value.shape == table.shape:
                    value[rows] = 0
            if table.grad is not None:
                table.grad[rows] = 0
            counts = torch.bincount(categories, minlength=4).tolist()
            self.total_proposed_rows += count
            events.append(self._event(epoch, step, "infuse", layer=name, rows=count,
                                     **dict(zip(("uniform", "identity", "inversion", "constant"), counts))))
        return events

    def state_dict(self):
        names = ("level", "stable_count", "last_accuracy", "last_loss", "best_accuracy",
                 "safe_accuracy", "safe_loss", "best_safe_accuracy", "last_transition_step",
                 "retry_after_step", "total_proposed_rows")
        return dict(version=1, options=dict(self.options), layers=list(self.tables),
                    **{name: getattr(self, name) for name in names})

    def load_state_dict(self, state):
        if (state.get("version") != 1 or state["options"] != self.options
                or state["layers"] != list(self.tables) or not 1 <= state["level"] <= 5):
            raise ValueError("Incompatible KD infusion state or layer layout.")
        for name in self.state_dict():
            if name not in {"version", "options", "layers"}:
                setattr(self, name, state[name])

    @torch.no_grad()
    def summary(self):
        constants = sum(int(((table >= 0).all(-1) | (table < 0).all(-1)).sum())
                        for table in self.tables.values())
        total = sum(table.shape[0] for table in self.tables.values())
        return dict(level=self.level, policy=asdict(self.policy), options=dict(self.options),
                    layers=list(self.tables), dense_nodes=total, constant_nodes=constants,
                    constant_fraction=constants / total, total_proposed_rows=self.total_proposed_rows,
                    stable_evaluations=self.stable_count, safe_accuracy=self.safe_accuracy,
                    best_safe_accuracy=self.best_safe_accuracy, rollback_accuracy_drop=0.015,
                    rollback_loss_increase=0.10)
