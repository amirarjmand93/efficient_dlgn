"""Adaptive five-level candidate infusion, independent of the training loop."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class InfusionLevel:
    level: int
    name: str
    uniform: float
    identity: float
    inversion: float
    constant: float

    def __post_init__(self):
        total = self.uniform + self.identity + self.inversion + self.constant
        if not math.isclose(total, 1.0, abs_tol=1e-12):
            raise ValueError(f"Infusion level {self.level} fractions sum to {total}, not 1.")

    @property
    def dense_mixture(self) -> tuple[float, float, float]:
        """CandidateLogic tuple; unlisted probability remains uniform."""
        return self.identity, self.inversion, self.constant


INFUSION_LEVELS = (
    InfusionLevel(1, "aggressive", 0.50, 0.30, 0.05, 0.15),
    InfusionLevel(2, "strong", 0.40, 0.25, 0.10, 0.25),
    InfusionLevel(3, "very_strong", 0.30, 0.20, 0.10, 0.40),
    InfusionLevel(4, "extreme", 0.25, 0.15, 0.10, 0.50),
    InfusionLevel(5, "maximum", 0.15, 0.10, 0.05, 0.70),
)
UNARY_BIASED_INFUSION_LEVELS = (
    InfusionLevel(1, "unary_biased_1", 0.50, 0.30, 0.05, 0.15),
    InfusionLevel(2, "unary_biased_2", 0.45, 0.30, 0.10, 0.15),
    InfusionLevel(3, "unary_biased_3", 0.40, 0.35, 0.10, 0.15),
    InfusionLevel(4, "unary_biased_4", 0.35, 0.35, 0.15, 0.15),
    InfusionLevel(5, "unary_biased_5", 0.30, 0.40, 0.15, 0.15),
)
INFUSION_SCHEDULES = {
    "aggressive": INFUSION_LEVELS,
    "unary_biased": UNARY_BIASED_INFUSION_LEVELS,
}


def infusion_levels(schedule: str) -> tuple[InfusionLevel, ...]:
    try:
        return INFUSION_SCHEDULES[schedule]
    except KeyError as error:
        raise ValueError("Infusion schedule must be aggressive or unary_biased.") from error


INFUSION_TRANSITION_FIELDS = (
    "step", "event", "from_level", "to_level", "validation_accuracy",
    "validation_loss", "safe_accuracy", "safe_loss", "reason", "checkpoint",
)


def apply_infusion_level(
    model, level: int, *, scope: str = "all", schedule: str = "aggressive",
) -> InfusionLevel:
    """Apply one level to a four-convolution, two-dense CompactLogic model."""
    from ..compact import CandidateLogicLayer

    if scope not in {"all", "conv", "dense"}:
        raise ValueError("Infusion scope must be all, conv, or dense.")
    levels = infusion_levels(schedule)
    try:
        policy = levels[int(level) - 1]
    except (IndexError, TypeError, ValueError) as error:
        raise ValueError(f"Infusion level must be from 1 to {len(levels)}.") from error
    if policy.level != level:
        raise ValueError(f"Infusion level must be from 1 to {len(levels)}.")
    layers = [module for module in model.modules() if isinstance(module, CandidateLogicLayer)]
    convolutions = [layer for layer in layers if layer.is_convolution]
    dense = [layer for layer in layers if not layer.is_convolution]
    if len(convolutions) != 4 or len(dense) != 2:
        raise ValueError("Progressive aggressive infusion requires four convolutions and two dense layers.")
    for layer in layers:
        layer.resample_preferred_ops = ()
    if scope in {"all", "conv"}:
        for layer in convolutions[:3]:
            layer.resample_preferred_ops = (1, 7)  # Same AND/OR prior as aggressive ELAR.
    for layer in dense:
        layer.resample_efficient_dense = policy.dense_mixture if scope in {"all", "dense"} else False
    return policy


@dataclass(frozen=True)
class InfusionDecision:
    action: str
    step: int
    from_level: int
    to_level: int
    accuracy: float
    loss: float
    reason: str


class ProgressiveAggressiveInfusion:
    """Advance on stable validation metrics and request rollback on regression.

    The class owns policy/state only. Callers decide how checkpoints are stored
    and restored, which keeps it reusable from other Python training scripts.
    """

    def __init__(
        self,
        model,
        *,
        exploration_steps: int,
        start_level: int = 1,
        scope: str = "all",
        schedule: str = "aggressive",
        level_1_only: bool = False,
        stable_evaluations: int = 3,
        minimum_transition_steps: int = 15_000,
        stable_accuracy_tolerance: float = 0.003,
        stable_loss_tolerance: float = 0.05,
        rollback_accuracy_drop: float = 0.015,
        rollback_loss_increase: float = 0.10,
    ):
        if exploration_steps <= 0 or stable_evaluations <= 0:
            raise ValueError("Exploration steps and stable evaluations must be positive.")
        levels = infusion_levels(schedule)
        if start_level not in range(1, len(levels) + 1):
            raise ValueError(f"Infusion start level must be from 1 to {len(levels)}.")
        if scope not in {"all", "conv", "dense"}:
            raise ValueError("Infusion scope must be all, conv, or dense.")
        if (scope == "conv" or level_1_only) and start_level != 1:
            raise ValueError("Conv-only or Level-1-only infusion requires start level 1.")
        if minimum_transition_steps < 0:
            raise ValueError("minimum_transition_steps must be nonnegative.")
        for name, value in (
            ("stable_accuracy_tolerance", stable_accuracy_tolerance),
            ("stable_loss_tolerance", stable_loss_tolerance),
            ("rollback_accuracy_drop", rollback_accuracy_drop),
            ("rollback_loss_increase", rollback_loss_increase),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative.")
        self.model = model
        self.exploration_steps = exploration_steps
        self.required_stable_evaluations = stable_evaluations
        self.minimum_transition_steps = minimum_transition_steps
        self.stable_accuracy_tolerance = stable_accuracy_tolerance
        self.stable_loss_tolerance = stable_loss_tolerance
        self.rollback_accuracy_drop = rollback_accuracy_drop
        self.rollback_loss_increase = rollback_loss_increase
        self.start_level = start_level
        self.scope = scope
        self.schedule = schedule
        self.level_1_only = bool(level_1_only)
        self.level = start_level
        self.stable_count = 0
        self.last_accuracy = None
        self.last_loss = None
        self.best_accuracy = None
        self.safe_level = None
        self.safe_accuracy = None
        self.best_safe_accuracy = None
        self.safe_loss = None
        self.last_transition_step = None
        apply_infusion_level(self.model, self.level, scope=self.scope, schedule=self.schedule)

    @property
    def policy(self) -> InfusionLevel:
        return infusion_levels(self.schedule)[self.level - 1]

    def observe(self, *, step: int, accuracy: float, loss: float) -> InfusionDecision:
        if not math.isfinite(accuracy) or not math.isfinite(loss) or not 0 <= accuracy <= 1:
            raise ValueError("Validation accuracy/loss must be finite and accuracy must be in [0,1].")
        if (self.safe_level is not None and self.level > self.safe_level
                and (accuracy < self.safe_accuracy - self.rollback_accuracy_drop
                     or loss > self.safe_loss + self.rollback_loss_increase)):
            return InfusionDecision(
                "rollback", step, self.level, self.safe_level, accuracy, loss,
                "validation metrics regressed beyond the safe checkpoint",
            )

        stable = (
            self.last_accuracy is None
            or (accuracy >= self.last_accuracy - self.stable_accuracy_tolerance
                and loss <= self.last_loss + self.stable_loss_tolerance)
        )
        self.best_accuracy = accuracy if self.best_accuracy is None else max(self.best_accuracy, accuracy)
        accuracy_reference = (
            self.best_safe_accuracy
            if self.best_safe_accuracy is not None else self.best_accuracy
        )
        acceptable = accuracy >= accuracy_reference - self.stable_accuracy_tolerance
        self.stable_count = self.stable_count + 1 if stable and acceptable else 0
        self.last_accuracy, self.last_loss = accuracy, loss
        cooldown_complete = (
            self.last_transition_step is None
            or step - self.last_transition_step >= self.minimum_transition_steps
        )
        if (self.stable_count >= self.required_stable_evaluations
                and self.level < len(infusion_levels(self.schedule))
                and self.scope != "conv" and not self.level_1_only
                and cooldown_complete
                and step < self.exploration_steps):
            return InfusionDecision(
                "advance", step, self.level, self.level + 1, accuracy, loss,
                f"{self.stable_count} stable acceptable validation evaluations",
            )
        if self.scope == "conv" or self.level_1_only:
            reason = "fixed Level 1 infusion policy"
        elif step >= self.exploration_steps:
            reason = "exploration complete"
        elif not cooldown_complete:
            reason = "waiting for transition cooldown"
        else:
            reason = "collecting stable evaluations"
        return InfusionDecision("hold", step, self.level, self.level, accuracy, loss, reason)

    def prepare_safe_point(self, *, step: int, accuracy: float, loss: float) -> None:
        """Record metrics that correspond to the checkpoint saved by the caller."""
        self.safe_level = self.level
        self.safe_accuracy = float(accuracy)
        self.best_safe_accuracy = (
            self.safe_accuracy if self.best_safe_accuracy is None
            else max(self.best_safe_accuracy, self.safe_accuracy)
        )
        self.safe_loss = float(loss)
        self.last_transition_step = int(step)

    def advance(self, decision: InfusionDecision) -> InfusionLevel:
        if decision.action != "advance" or decision.from_level != self.level:
            raise ValueError("Decision does not advance the current infusion level.")
        self.level = decision.to_level
        self.stable_count = 0
        self.last_transition_step = decision.step
        return apply_infusion_level(self.model, self.level, scope=self.scope, schedule=self.schedule)

    def reset_after_rollback(self, *, step: int) -> None:
        self.stable_count = 0
        self.last_transition_step = int(step)
        apply_infusion_level(self.model, self.level, scope=self.scope, schedule=self.schedule)

    def summary(self) -> dict:
        return {
            "level": self.level,
            "start_level": self.start_level,
            "scope": self.scope,
            "schedule": self.schedule,
            "level_1_only": self.level_1_only,
            "name": self.policy.name,
            "fractions": asdict(self.policy),
            "stable_evaluations": self.stable_count,
            "required_stable_evaluations": self.required_stable_evaluations,
            "minimum_transition_steps": self.minimum_transition_steps,
            "rollback_accuracy_drop": self.rollback_accuracy_drop,
            "rollback_loss_increase": self.rollback_loss_increase,
            "safe_level": self.safe_level,
            "safe_accuracy": self.safe_accuracy,
            "best_safe_accuracy": self.best_safe_accuracy,
            "safe_loss": self.safe_loss,
            "last_transition_step": self.last_transition_step,
        }

    def state_dict(self) -> dict:
        return {
            "version": 1,
            "start_level": self.start_level,
            "scope": self.scope,
            "schedule": self.schedule,
            "level_1_only": self.level_1_only,
            "level": self.level,
            "stable_count": self.stable_count,
            "last_accuracy": self.last_accuracy,
            "last_loss": self.last_loss,
            "best_accuracy": self.best_accuracy,
            "safe_level": self.safe_level,
            "safe_accuracy": self.safe_accuracy,
            "best_safe_accuracy": self.best_safe_accuracy,
            "safe_loss": self.safe_loss,
            "last_transition_step": self.last_transition_step,
        }

    def load_state_dict(self, state: dict) -> None:
        if state.get("version") != 1 or int(state.get("level", 0)) not in range(1, len(infusion_levels(self.schedule)) + 1):
            raise ValueError("Unsupported progressive infusion checkpoint state.")
        if int(state.get("start_level", 1)) != self.start_level:
            raise ValueError("Infusion start level differs from the checkpoint; pass its original --compact-infusion-start-level.")
        if state.get("scope", "all") != self.scope or bool(state.get("level_1_only", False)) != self.level_1_only:
            raise ValueError("Infusion scope or Level-1-only setting differs from the checkpoint.")
        if state.get("schedule", "aggressive") != self.schedule:
            raise ValueError("Infusion schedule differs from the checkpoint.")
        self.level = int(state["level"])
        self.stable_count = int(state["stable_count"])
        self.last_accuracy = state["last_accuracy"]
        self.last_loss = state["last_loss"]
        self.best_accuracy = state["best_accuracy"]
        self.safe_level = state["safe_level"]
        self.safe_accuracy = state["safe_accuracy"]
        self.best_safe_accuracy = state.get("best_safe_accuracy", self.safe_accuracy)
        self.safe_loss = state["safe_loss"]
        self.last_transition_step = state["last_transition_step"]
        apply_infusion_level(self.model, self.level, scope=self.scope, schedule=self.schedule)
