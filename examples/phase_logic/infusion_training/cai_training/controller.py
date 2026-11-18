"""Validation-controlled candidate infusion strength for the CAI experiment."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class CAIDecision:
    action: str
    reason: str


class ContinuousAdaptiveInfusion:
    """Adapt future resampling probabilities using elapsed steps and validation accuracy.

    step_up/down specify fractions of the remaining distance to 1/0 per
    reference interval (5% of exploration), not fixed alpha increments.
    Time-scaled EMA and smoothing make the response comparable across evaluation
    frequencies. Alpha changes cannot undo earlier rewiring.
    """

    def __init__(self, policy, *, exploration_steps: int, initial_alpha: float = 0.7,
                 step_up: float = 0.08, step_down: float = 0.15,
                 step_down_mode: str = "fractional",
                 smoothing: float = 0.5, ema_beta: float = 0.7,
                 accuracy_tolerance: float = 0.003, rollback_drop: float = 0.015,
                 rollback_confirmations: int = 3):
        values = (initial_alpha, step_up, step_down, smoothing, ema_beta,
                  accuracy_tolerance, rollback_drop)
        if exploration_steps <= 0 or any(not math.isfinite(value) for value in values):
            raise ValueError("CAI feedback settings must be finite and exploration positive.")
        if not 0 <= initial_alpha <= 1 or not 0 < step_up < 1 or not 0 < step_down < 1:
            raise ValueError("CAI alpha and response fractions are out of range.")
        if step_down_mode not in {"fractional", "absolute"}:
            raise ValueError("CAI step-down mode must be fractional or absolute.")
        if not 0 < smoothing <= 1 or not 0 <= ema_beta < 1:
            raise ValueError("CAI smoothing or EMA beta is out of range.")
        if not 0 <= accuracy_tolerance <= 1 or not 0 <= rollback_drop <= 1:
            raise ValueError("CAI accuracy tolerances must be in [0, 1].")
        if (not isinstance(rollback_confirmations, int)
                or isinstance(rollback_confirmations, bool)
                or rollback_confirmations <= 0):
            raise ValueError("CAI rollback confirmations must be a positive integer.")
        self.policy = policy
        self.exploration_steps = exploration_steps
        self.initial_alpha = float(initial_alpha)
        self.step_up = float(step_up)
        self.step_down = float(step_down)
        self.step_down_mode = step_down_mode
        self.smoothing = float(smoothing)
        self.ema_beta = float(ema_beta)
        self.accuracy_tolerance = float(accuracy_tolerance)
        self.rollback_drop = float(rollback_drop)
        self.rollback_confirmations = rollback_confirmations
        self.u = float(initial_alpha)
        self.alpha = float(initial_alpha)
        self.accuracy_ema = None
        self.best_accuracy_ema = None
        self.last_observation_step = 0
        self.safe_accuracy = None
        self.safe_alpha = None
        self.rollback_bad_count = 0
        self.policy.set_alpha(self.alpha)

    def _time_factor(self, fraction: float, per_interval_fraction: float) -> float:
        # The reference interval is 1/20 of exploration. This composition is
        # invariant to subdividing an unchanged observation interval.
        return 1.0 - (1.0 - per_interval_fraction) ** (20.0 * fraction)

    def _smooth_alpha(self, fraction: float):
        factor = self._time_factor(fraction, self.smoothing)
        self.alpha += factor * (self.u - self.alpha)
        self.alpha = min(1.0, max(0.0, self.alpha))
        self.policy.set_alpha(self.alpha)

    def observe(self, *, step: int, accuracy: float, loss: float) -> CAIDecision:
        if not math.isfinite(accuracy) or not 0 <= accuracy <= 1 or not math.isfinite(loss):
            raise ValueError("CAI validation accuracy and loss must be finite.")
        if step <= self.last_observation_step:
            raise ValueError("CAI observations must have increasing training steps.")
        elapsed = min(step, self.exploration_steps) - min(self.last_observation_step, self.exploration_steps)
        fraction = elapsed / self.exploration_steps
        if self.accuracy_ema is None:
            self.accuracy_ema = accuracy
        else:
            decay = self.ema_beta ** (20.0 * fraction)
            self.accuracy_ema = decay * self.accuracy_ema + (1.0 - decay) * accuracy
        self.best_accuracy_ema = max(
            self.accuracy_ema,
            self.best_accuracy_ema if self.best_accuracy_ema is not None else self.accuracy_ema,
        )
        self.last_observation_step = step
        if elapsed <= 0:
            return CAIDecision("hold", "exploration complete")
        rollback_poor = (
            self.safe_accuracy is not None and self.safe_alpha is not None
            and accuracy < self.safe_accuracy - self.rollback_drop
        )
        if rollback_poor:
            self.rollback_bad_count += 1
            if self.rollback_bad_count >= self.rollback_confirmations:
                return CAIDecision(
                    "rollback",
                    f"validation accuracy failed {self.rollback_bad_count} consecutive checks",
                )
            return CAIDecision(
                "hold",
                f"rollback confirmation {self.rollback_bad_count}/{self.rollback_confirmations}",
            )
        self.rollback_bad_count = 0
        previous_alpha = self.alpha
        if self.accuracy_ema < self.best_accuracy_ema - self.accuracy_tolerance:
            if self.step_down_mode == "absolute":
                self.u = max(0.0, self.u - self.step_down)
            else:
                self.u *= 1.0 - self._time_factor(fraction, self.step_down)
            action = "decrease"
            reason = "validation accuracy below best smoothed reference"
        else:
            self.u += (1.0 - self.u) * self._time_factor(fraction, self.step_up)
            action = "increase"
            reason = "validation accuracy healthy"
        self._smooth_alpha(fraction)
        if math.isclose(self.alpha, previous_alpha, abs_tol=1e-12):
            return CAIDecision("hold", "infusion strength at limit")
        return CAIDecision(action, reason)

    def prepare_safe_point(self, *, step: int, accuracy: float):
        self.safe_accuracy = float(accuracy)
        self.safe_alpha = self.alpha

    def recover_after_rollback(self, *, step: int):
        # The checkpoint may be older than the failed evaluation; do not count
        # that elapsed training time again on the next observation.
        self.last_observation_step = step
        if self.step_down_mode == "absolute":
            self.u = max(0.0, self.u - self.step_down)
        else:
            self.u *= 1.0 - self.step_down
        self._smooth_alpha(1.0 / 20.0)
        self.safe_accuracy = None
        self.safe_alpha = None
        self.rollback_bad_count = 0

    def state_dict(self) -> dict:
        return {
            "version": 3, "u": self.u, "alpha": self.alpha,
            "accuracy_ema": self.accuracy_ema, "best_accuracy_ema": self.best_accuracy_ema,
            "last_observation_step": self.last_observation_step,
            "safe_accuracy": self.safe_accuracy, "safe_alpha": self.safe_alpha,
            "rollback_bad_count": self.rollback_bad_count,
        }

    def load_state_dict(self, state: dict):
        version = state.get("version")
        required = {
            "u", "alpha", "accuracy_ema", "best_accuracy_ema",
            "last_observation_step", "safe_accuracy", "safe_alpha",
        }
        if version not in {2, 3} or not required.issubset(state):
            raise ValueError("Unsupported CAI controller state.")
        if not 0 <= state["u"] <= 1 or not 0 <= state["alpha"] <= 1:
            raise ValueError("CAI checkpoint has invalid infusion strength.")
        for key in required:
            setattr(self, key, state[key])
        self.rollback_bad_count = state.get("rollback_bad_count", 0)
        self.policy.set_alpha(self.alpha)
