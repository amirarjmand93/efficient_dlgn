'''"""Classification and phase regularization losses."""

from dataclasses import dataclass

import torch

from torchlogix.layers import LogicConv2d, LogicDense
from torchlogix.parametrization import LightLUTParametrization


LOSS_OPTIONS = (
    "cross_entropy",
    "cross_entropy_phase",
)


def _lightlut_weights(module: torch.nn.Module):
    if isinstance(module, LogicDense):
        yield module.weight
    elif isinstance(module, LogicConv2d):
        yield from module.tree_weights


def margin_phase_statistics(
    model: torch.nn.Module, margin: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return mean margin energy and fraction of logits outside the margin."""

    total = None
    count = 0
    outside = None
    for module in model.modules():
        parametrization = getattr(module, "parametrization", None)
        if not isinstance(parametrization, LightLUTParametrization):
            continue

        for weight in _lightlut_weights(module):
            energy = torch.relu(margin - weight.abs()).square()
            confident = (weight.abs() >= margin).sum()
            total = energy.sum() if total is None else total + energy.sum()
            outside = confident if outside is None else outside + confident
            count += energy.numel()

    if total is None or outside is None or count == 0:
        raise RuntimeError("The model contains no LightLUT parameters.")
    return total / count, outside.to(total.dtype) / count


def margin_phase_energy(
    model: torch.nn.Module, margin: float = 1.0
) -> torch.Tensor:
    """Mean squared shortfall of raw LightLUT logits from ``margin``."""

    energy, _ = margin_phase_statistics(model, margin)
    return energy


@dataclass
class LossTerms:
    total: torch.Tensor
    cross_entropy: torch.Tensor
    phase: torch.Tensor


class PureCrossEntropyLoss(torch.nn.CrossEntropyLoss):
    """Explicit name for the unmodified supervised CE reference loss."""


class PhaseLogicLoss(torch.nn.Module):
    """Select pure CE or CE with a weighted logit-margin phase term."""

    def __init__(
        self,
        loss_option: str,
        margin: float = 1.0,
    ):
        super().__init__()
        if loss_option not in LOSS_OPTIONS:
            raise ValueError(
                f"Unknown loss option {loss_option!r}; choose from {LOSS_OPTIONS}."
            )
        self.loss_option = loss_option
        self.margin = margin
        self.cross_entropy = PureCrossEntropyLoss()

    def forward(
        self,
        model: torch.nn.Module,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        phase_weight: float,
    ) -> LossTerms:
        ce = self.cross_entropy(predictions, targets)
        if self.loss_option == "cross_entropy_phase":
            phase = margin_phase_energy(model, self.margin)
            total = ce + phase_weight * phase
        else:
            phase = torch.zeros((), device=ce.device, dtype=ce.dtype)
            total = ce
        return LossTerms(
            total=total,
            cross_entropy=ce,
            phase=phase,
        )
'''


"""Classification and Allen-Cahn double-well phase regularization losses."""

from dataclasses import dataclass

import torch

from torchlogix.layers import LogicConv2d, LogicDense
from torchlogix.parametrization import LightLUTParametrization


LOSS_OPTIONS = (
    "cross_entropy",
    "cross_entropy_phase",
)


def _lightlut_weights(module: torch.nn.Module):
    if isinstance(module, LogicDense):
        yield module.weight
    elif isinstance(module, LogicConv2d):
        yield from module.tree_weights


def margin_phase_statistics(
    model: torch.nn.Module,
    margin: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Return mean Allen-Cahn double-well energy and fraction of logits
    whose magnitude has reached the target well magnitude.

    The two stable phases are -margin and +margin.
    """
    if margin <= 0:
        raise ValueError("margin must be positive.")

    total = None
    count = 0
    outside = None

    for module in model.modules():
        parametrization = getattr(module, "parametrization", None)

        if not isinstance(parametrization, LightLUTParametrization):
            continue

        for weight in _lightlut_weights(module):
            # Normalized Allen-Cahn double-well potential:
            #
            #     W(u) = 1/4 (u^2 - 1)^2
            #
            # with u = weight / margin.
            #
            # Therefore the stable wells are at:
            #     weight = -margin
            #     weight = +margin
            normalized = weight / margin
            energy = 0.25 * (normalized.square() - 1.0).square()

            reached_well = (weight.abs() >= margin).sum()

            total = (
                energy.sum()
                if total is None
                else total + energy.sum()
            )

            outside = (
                reached_well
                if outside is None
                else outside + reached_well
            )

            count += energy.numel()

    if total is None or outside is None or count == 0:
        raise RuntimeError("The model contains no LightLUT parameters.")

    mean_energy = total / count
    fraction_outside = outside.to(total.dtype) / count

    return mean_energy, fraction_outside


def margin_phase_energy(
    model: torch.nn.Module,
    margin: float = 1.0,
) -> torch.Tensor:
    """
    Mean Allen-Cahn double-well potential over raw LightLUT logits.

    Stable phases are at -margin and +margin.
    """
    energy, _ = margin_phase_statistics(model, margin)
    return energy


@dataclass
class LossTerms:
    total: torch.Tensor
    cross_entropy: torch.Tensor
    phase: torch.Tensor


class PureCrossEntropyLoss(torch.nn.CrossEntropyLoss):
    """Unmodified supervised cross-entropy loss."""


class PhaseLogicLoss(torch.nn.Module):
    """
    Cross entropy with optional Allen-Cahn double-well regularization.

    Total loss:

        L = L_CE + phase_weight * L_phase

    where

        L_phase = mean[ 1/4 * ((w / margin)^2 - 1)^2 ]

    The two stable phases are w = -margin and w = +margin.
    """

    def __init__(
        self,
        loss_option: str,
        margin: float = 1.0,
    ):
        super().__init__()

        if loss_option not in LOSS_OPTIONS:
            raise ValueError(
                f"Unknown loss option {loss_option!r}; "
                f"choose from {LOSS_OPTIONS}."
            )

        if margin <= 0:
            raise ValueError("margin must be positive.")

        self.loss_option = loss_option
        self.margin = margin
        self.cross_entropy = PureCrossEntropyLoss()

    def forward(
        self,
        model: torch.nn.Module,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        phase_weight: float,
    ) -> LossTerms:

        ce = self.cross_entropy(predictions, targets)

        if self.loss_option == "cross_entropy_phase":
            phase = margin_phase_energy(model, self.margin)
            total = ce + phase_weight * phase
        else:
            phase = torch.zeros(
                (),
                device=ce.device,
                dtype=ce.dtype,
            )
            total = ce

        return LossTerms(
            total=total,
            cross_entropy=ce,
            phase=phase,
        )