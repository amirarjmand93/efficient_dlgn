"""Configuration for the MNIST phase-logic comparisons."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


EXPERIMENTS = (
    "pure_ce",
    "annealing",
    "annealing_phase",
)
PARAMETRIZATIONS = ("light", "raw", "warp")
LR_SCHEDULES = ("constant", "cosine")
MODEL_VARIANTS_BY_DATASET = {
    "mnist": (
        "build_model",
        "build_model_linear_head",
    ),
    "cifar10": (
        "build_model_cifar10",
        "build_model_cifar10_org",
        "build_model_cifar10_smallmodel",
        "build_model_cifar10_small_modifieddense",
        "build_model_cifar10_small_classrouted",
        "build_model_cifar10_small_classrouted_channelmixed",
        "build_model_cifar10_deepconv",
        "build_model_cifar10_linear_head",
    ),
    "cifar100": (
        "build_model_cifar100_small_classrouted",
        "build_model_cifar100_small_classrouted_channelmixed",
    ),
}
MODEL_VARIANTS = tuple(
    model_name
    for dataset_models in MODEL_VARIANTS_BY_DATASET.values()
    for model_name in dataset_models
)
DEFAULT_MODEL_VARIANT = {
    "mnist": "build_model",
    "cifar10": "build_model_cifar10",
    "cifar100": "build_model_cifar100_small_classrouted",
}
DATASETS = ("mnist", "cifar10", "cifar100")
INPUT_ENCODINGS = ("fixed", "quantile", "sinusoidal", "hybrid")
INPUT_BIT_OPTIONS = (1, 2, 4)


@dataclass(frozen=True)
class ExperimentConfig:
    """All settings needed to reproduce one experiment run."""

    experiment: str = "annealing"
    parametrization: str = "light"
    model_variant: str = "build_model"
    dataset: str = "mnist"
    seed: int = 42
    device: str = "auto"
    data_dir: Path = Path("data")
    output_dir: Path = Path("examples/phase_logic/results")
    input_encoding: str = "fixed"
    num_input_bits: int = 1
    num_quantile_bits: int | None = None
    num_sine_bits: int | None = None
    sine_sigma: float = 1.0
    sine_seed: int = 0
    cifar_augmentation: bool = False
    cifar_random_erasing: bool = False

    iterations: int = 1_000
    eval_frequency: int = 100
    batch_size: int = 128
    learning_rate: float = 0.02
    lr_schedule: str = "constant"
    end_learning_rate: float = 0.002
    num_workers: int = 0

    start_temperature: float = 1.0
    end_temperature: float = 0.3
    temperature_warmup: int = 200

    phase_margin: float = 1.0
    phase_lambda_max: float = 0.2
    phase_warmup: int = 200
    save_model: bool = False

    @property
    def uses_temperature_annealing(self) -> bool:
        return self.experiment in {
            "annealing",
            "annealing_phase",
        }

    @property
    def uses_phase_loss(self) -> bool:
        return self.experiment in {
            "annealing_phase",
        }

    @property
    def loss_option(self) -> str:
        return "cross_entropy_phase" if self.uses_phase_loss else "cross_entropy"

    @property
    def run_dir(self) -> Path:
        path = self.output_dir / self.dataset
        if self.dataset in {"cifar10", "cifar100"}:
            path = path / self.input_encoding_slug
            if self.cifar_augmentation:
                path = path / "crop_flip_augmentation"
                if self.cifar_random_erasing:
                    path = path / "random_erasing"
        if self.lr_schedule == "cosine":
            path = path / self.lr_schedule_slug
        return (
            path
            / self.parametrization
            / self.model_variant
            / self.experiment
            / f"seed_{self.seed}"
        )

    @property
    def input_encoding_slug(self) -> str:
        if self.input_encoding in {"fixed", "quantile"}:
            return f"{self.input_encoding}_{self.quantile_bits}bit"
        sine_suffix = (
            f"sigma_{self.sine_sigma:g}_seed_{self.sine_seed}"
            .replace(".", "p")
        )
        if self.input_encoding == "sinusoidal":
            return f"sinusoidal_{self.sine_bits}bit_{sine_suffix}"
        return (
            f"hybrid_q{self.quantile_bits}_s{self.sine_bits}_"
            f"{sine_suffix}"
        )

    @property
    def lr_schedule_slug(self) -> str:
        """Filesystem-safe description of a non-constant LR schedule."""

        start = f"{self.learning_rate:g}".replace(".", "p")
        end = f"{self.end_learning_rate:g}".replace(".", "p")
        return f"cosine_lr_{start}_to_{end}"

    @property
    def quantile_bits(self) -> int:
        if self.input_encoding == "fixed":
            return 1
        if self.input_encoding == "quantile":
            return self.num_quantile_bits or self.num_input_bits
        if self.input_encoding == "hybrid":
            return self.num_quantile_bits or 2
        return 0

    @property
    def sine_bits(self) -> int:
        if self.input_encoding == "sinusoidal":
            return self.num_sine_bits or self.num_input_bits
        if self.input_encoding == "hybrid":
            return self.num_sine_bits or 2
        return 0

    @property
    def total_input_bits(self) -> int:
        return self.quantile_bits + self.sine_bits

    def validate(self) -> None:
        if self.experiment not in EXPERIMENTS:
            raise ValueError(
                f"Unknown experiment {self.experiment!r}; choose from {EXPERIMENTS}."
            )
        if self.parametrization not in PARAMETRIZATIONS:
            raise ValueError(
                f"Unknown parametrization {self.parametrization!r}; "
                f"choose from {PARAMETRIZATIONS}."
            )
        if self.model_variant not in MODEL_VARIANTS:
            raise ValueError(
                f"Unknown model variant {self.model_variant!r}; "
                f"choose from {MODEL_VARIANTS}."
            )
        if self.dataset not in DATASETS:
            raise ValueError(
                f"Unknown dataset {self.dataset!r}; choose from {DATASETS}."
            )
        if self.model_variant not in MODEL_VARIANTS_BY_DATASET[self.dataset]:
            raise ValueError(
                f"Model builder {self.model_variant!r} is not compatible with "
                f"dataset {self.dataset!r}; choose from "
                f"{MODEL_VARIANTS_BY_DATASET[self.dataset]}."
            )
        self._validate_input_encoding()
        positive_values = {
            "iterations": self.iterations,
            "eval_frequency": self.eval_frequency,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "start_temperature": self.start_temperature,
            "end_temperature": self.end_temperature,
        }
        for name, value in positive_values.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.lr_schedule not in LR_SCHEDULES:
            raise ValueError(
                f"Unknown lr_schedule {self.lr_schedule!r}; "
                f"choose from {LR_SCHEDULES}."
            )
        if self.lr_schedule == "cosine":
            if self.end_learning_rate < 0:
                raise ValueError("end_learning_rate cannot be negative")
            if self.end_learning_rate > self.learning_rate:
                raise ValueError("end_learning_rate cannot exceed learning_rate")
        warmups = {
            "temperature_warmup": self.temperature_warmup,
            "phase_warmup": self.phase_warmup,
        }
        for name, value in warmups.items():
            if not 0 <= value < self.iterations:
                raise ValueError(f"{name} must be in [0, iterations)")
        self._validate_phase_settings()

    def _validate_input_encoding(self) -> None:
        if self.input_encoding not in INPUT_ENCODINGS:
            raise ValueError(
                f"Unknown input encoding {self.input_encoding!r}; "
                f"choose from {INPUT_ENCODINGS}."
            )
        if self.num_input_bits not in INPUT_BIT_OPTIONS:
            raise ValueError(
                f"num_input_bits must be one of {INPUT_BIT_OPTIONS}, "
                f"got {self.num_input_bits}."
            )
        for name, value in (
            ("num_quantile_bits", self.num_quantile_bits),
            ("num_sine_bits", self.num_sine_bits),
        ):
            if value is not None and value not in INPUT_BIT_OPTIONS:
                raise ValueError(
                    f"{name} must be one of {INPUT_BIT_OPTIONS}, got {value}."
                )
        if self.sine_sigma <= 0:
            raise ValueError("sine_sigma must be positive.")
        if self.dataset == "mnist":
            if self.cifar_augmentation or self.cifar_random_erasing:
                raise ValueError("CIFAR augmentation cannot be used with MNIST.")
            if self.input_encoding != "fixed" or self.num_input_bits != 1:
                raise ValueError(
                    "MNIST currently requires input_encoding='fixed' and "
                    "num_input_bits=1."
                )
            return
        if self.cifar_random_erasing and not self.cifar_augmentation:
            raise ValueError(
                "cifar_random_erasing requires cifar_augmentation so it is "
                "applied with crop and flip."
            )
        if self.input_encoding == "fixed" and self.num_input_bits != 1:
            raise ValueError("Fixed CIFAR encoding requires num_input_bits=1.")
        if self.input_encoding == "quantile" and self.quantile_bits not in (2, 4):
            raise ValueError(
                "Quantile CIFAR encoding requires 2 or 4 quantile bits."
            )
        if self.input_encoding == "sinusoidal" and self.sine_bits not in (1, 2, 4):
            raise ValueError("Sinusoidal CIFAR encoding requires 1, 2, or 4 bits.")
        if self.input_encoding == "hybrid":
            if self.quantile_bits not in (2, 4):
                raise ValueError("Hybrid encoding requires 2 or 4 quantile bits.")
            if self.sine_bits not in (1, 2, 4):
                raise ValueError("Hybrid encoding requires 1, 2, or 4 sine bits.")

    def _validate_phase_settings(self) -> None:
        if self.uses_phase_loss and self.parametrization != "light":
            raise ValueError("phase experiments require parametrization='light'")
        if self.phase_margin <= 0:
            raise ValueError("phase_margin must be positive")
        if self.phase_lambda_max < 0:
            raise ValueError("phase_lambda_max cannot be negative")

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["data_dir"] = str(self.data_dir)
        values["output_dir"] = str(self.output_dir)
        return values
