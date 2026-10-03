"""Controlled LUT parametrization and phase-logic experiments for MNIST."""

from .config import ExperimentConfig
from .training import run_experiment

__all__ = ["ExperimentConfig", "run_experiment"]
