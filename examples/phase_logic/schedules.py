"""Learning-rate, temperature, and regularization schedules."""

import math
import torch

from torchlogix.parametrization import LUTParametrization


def linear_after_warmup(
    step: int,
    total_steps: int,
    warmup_steps: int,
    start: float,
    end: float,
) -> float:
    """Hold ``start`` through warm-up, then linearly move to ``end``."""

    if step <= warmup_steps:
        return start
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    return start + progress * (end - start)


def cosine_decay(
    step: int,
    total_steps: int,
    start: float,
    end: float,
) -> float:
    """Cosine-decay ``start`` at step 1 to ``end`` at the final step."""

    if total_steps <= 1:
        return end
    progress = (step - 1) / (total_steps - 1)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return end + (start - end) * cosine


def set_parametrization_temperature(
    model: torch.nn.Module, temperature: float
) -> None:
    """Apply a temperature to every LUT parametrization in a model."""

    found = False
    for module in model.modules():
        parametrization = getattr(module, "parametrization", None)
        if isinstance(parametrization, LUTParametrization):
            parametrization.update_temperature(temperature)
            found = True
    if not found:
        raise RuntimeError("The model contains no LUT parametrizations.")
