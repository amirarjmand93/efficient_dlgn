"""Low-overhead selected-gate stability metrics for CAI validation checkpoints."""

from collections import OrderedDict
from pathlib import Path

import torch

from ...compact import CandidateLogicLayer


CONSTANT_OPS = (0, 15)
UNARY_OPS = (3, 5, 10, 12)
TRANSITION_FIELDS = (
    "selected_gate_changes",
    "selected_gate_change_rate",
    "eligible_gate_change_rate",
    "gate_simplification_rate",
    "gate_complexification_rate",
)
COMPOSITION_FIELDS = (
    "unary_percent",
    "two_input_percent",
    "constant_percent",
)
GATE_METRIC_FIELDS = TRANSITION_FIELDS + COMPOSITION_FIELDS


def _stage_groups(model) -> OrderedDict[str, list[CandidateLogicLayer]]:
    """Return traversal-ordered Conv and logical Dense stages.

    Class-routed dense engines are grouped by their parent route stage so their
    statistics remain Dense1/Dense2 rather than one column set per class.
    """

    candidates = [
        (name, module) for name, module in model.named_modules()
        if isinstance(module, CandidateLogicLayer)
    ]
    groups = OrderedDict()
    conv_index = 0
    dense_keys = OrderedDict()
    for name, layer in candidates:
        if layer.is_convolution:
            conv_index += 1
            groups[f"conv{conv_index}"] = [layer]
        else:
            key = name.split(".routes.", 1)[0]
            dense_keys.setdefault(key, []).append(layer)
    for dense_index, engines in enumerate(dense_keys.values(), start=1):
        groups[f"dense{dense_index}"] = engines
    if not groups:
        raise ValueError("Gate stability tracking requires CompactLogic candidate layers.")
    return groups


def _selected_operations(layer: CandidateLogicLayer) -> torch.Tensor:
    """Copy only one selected uint8 operation per output row to CPU."""

    with torch.no_grad():
        winner = layer.weights.detach().argmax(dim=-1)
        sequence = layer.impl.gate_sequence.detach()
        selected = sequence.gather(1, winner[:, None]).squeeze(1)
        return selected.to(device="cpu", dtype=torch.uint8)


def _category_masks(operations: torch.Tensor):
    constant = (operations == CONSTANT_OPS[0]) | (operations == CONSTANT_OPS[1])
    unary = torch.zeros_like(constant)
    for operation in UNARY_OPS:
        unary |= operations == operation
    two_input = ~(constant | unary)
    return unary, two_input, constant


class GateStabilityTracker:
    """Compare discrete gate selections at consecutive validation checkpoints."""

    def __init__(self, model):
        self.groups = _stage_groups(model)
        self.previous = None

    @property
    def layer_names(self):
        return tuple(self.groups)

    @property
    def metric_fields(self):
        fields = list(GATE_METRIC_FIELDS)
        for layer_name in self.layer_names:
            fields.extend(f"{layer_name}_{field}" for field in GATE_METRIC_FIELDS)
        return tuple(fields)

    def _snapshot(self):
        selected = OrderedDict()
        eligible = OrderedDict()
        for name, engines in self.groups.items():
            chunks = [_selected_operations(engine) for engine in engines]
            selected[name] = torch.cat(chunks)
            eligible[name] = torch.cat([
                torch.full((chunk.numel(),), not engine.is_frozen, dtype=torch.bool)
                for chunk, engine in zip(chunks, engines)
            ])
        return selected, eligible

    @staticmethod
    def _statistics(current, previous, eligible):
        total = current.numel()
        unary, two_input, constant = _category_masks(current)
        if previous is None:
            changes = simplifications = complexifications = 0
            eligible_changes = 0
        else:
            changed = current != previous
            previous_unary, previous_two_input, previous_constant = _category_masks(previous)
            previous_simple = previous_unary | previous_constant
            current_simple = unary | constant
            changes = int(changed.sum())
            eligible_changes = int((changed & eligible).sum())
            simplifications = int((previous_two_input & current_simple).sum())
            complexifications = int((previous_simple & two_input).sum())
        eligible_total = int(eligible.sum())
        return {
            "selected_gate_changes": changes,
            "selected_gate_change_rate": changes / total if total else 0.0,
            "eligible_gate_change_rate": (
                eligible_changes / eligible_total if eligible_total else 0.0
            ),
            "gate_simplification_rate": simplifications / total if total else 0.0,
            "gate_complexification_rate": complexifications / total if total else 0.0,
            "unary_percent": 100.0 * int(unary.sum()) / total if total else 0.0,
            "two_input_percent": 100.0 * int(two_input.sum()) / total if total else 0.0,
            "constant_percent": 100.0 * int(constant.sum()) / total if total else 0.0,
        }

    def measure(self):
        current, eligible = self._snapshot()
        previous = self.previous
        layer_statistics = OrderedDict()
        for name in self.layer_names:
            layer_statistics[name] = self._statistics(
                current[name], None if previous is None else previous[name], eligible[name]
            )
        all_current = torch.cat(tuple(current.values()))
        all_previous = None if previous is None else torch.cat(tuple(previous.values()))
        all_eligible = torch.cat(tuple(eligible.values()))
        metrics = self._statistics(all_current, all_previous, all_eligible)
        for name, statistics in layer_statistics.items():
            metrics.update({f"{name}_{field}": value for field, value in statistics.items()})
        self.previous = current
        return metrics


def plot_gate_dynamics(rows: list[dict], output: Path) -> bool:
    """Write CAI gate-change, simplification, and alpha curves after training."""

    if not rows:
        return False
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    steps = [int(row["step"]) for row in rows]
    figure, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(steps, [100.0 * row["selected_gate_change_rate"] for row in rows])
    axes[0].set_ylabel("Gate changes (%)")
    axes[1].plot(steps, [100.0 * row["gate_simplification_rate"] for row in rows],
                 label="two-input → simple")
    axes[1].plot(steps, [100.0 * row["gate_complexification_rate"] for row in rows],
                 label="simple → two-input", alpha=0.75)
    axes[1].set_ylabel("Transitions (%)")
    axes[1].legend(loc="best")
    axes[2].plot(steps, [float(row["alpha"]) for row in rows])
    axes[2].set_ylabel("Infusion α")
    axes[2].set_xlabel("Training step")
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.suptitle("CAI selected-gate dynamics")
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return True
