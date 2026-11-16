"""Progressive gate-candidate infusion for CompactLogic training."""

from .controller import (
    INFUSION_LEVELS,
    UNARY_BIASED_INFUSION_LEVELS,
    INFUSION_TRANSITION_FIELDS,
    InfusionDecision,
    ProgressiveAggressiveInfusion,
    apply_infusion_level,
    infusion_levels,
)
from .checkpoint import load_infusion_checkpoint, save_infusion_checkpoint

__all__ = (
    "INFUSION_LEVELS",
    "UNARY_BIASED_INFUSION_LEVELS",
    "INFUSION_TRANSITION_FIELDS",
    "InfusionDecision",
    "ProgressiveAggressiveInfusion",
    "apply_infusion_level",
    "infusion_levels",
    "load_infusion_checkpoint",
    "save_infusion_checkpoint",
)
