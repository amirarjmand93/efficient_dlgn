"""Configuration for the MNIST phase-logic comparisons."""

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any


EXPERIMENTS = (
    "pure_ce",
    "annealing",
    "annealing_phase",
)
PARAMETRIZATIONS = ("light", "raw", "warp", "compact")
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
        "build_model_cifar10_compact_small",
        "build_model_cifar10_compact_sixconv",
        "build_model_cifar10_small_modifieddense",
        "build_model_cifar10_small_classrouted",
        "build_model_cifar10_small_classrouted_compact_head",
        "build_model_cifar10_small_compact_head",
        "build_model_cifar10_small_compact_head_wide75",
        "build_model_cifar10_small_mrlb_compact_head",
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
INPUT_ENCODINGS = ("fixed", "quantile", "sinusoidal", "hybrid", "thermometer")
INPUT_BIT_OPTIONS = (1, 2, 4)
THERMOMETER_BIT_OPTIONS = (1, 2, 3, 4, 8)


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
    hybrid_features: str = "sine"
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
    rewire_loss_probe: bool = False
    rewire_warmup: int = 10_000
    rewire_interval: int = 1_000
    rewire_max_rows: int = 8
    rewire_candidates: int = 8
    rewire_feature_pool: int = 256
    rewire_entropy_threshold: float = 0.15
    rewire_entropy_delta: float = 0.002
    rewire_min_improvement: float = 0.0
    compact_backend: str = "auto"
    mrlb_conv3_relational_channels: int = 128
    compact_exploration_steps: int = 400_000
    compact_resample_patience: int = 100
    compact_layer_aware_resampling: bool = False
    compact_efficient_layer_aware_resampling: bool = False
    compact_aggressive_layer_aware_resampling: bool = False
    compact_freeze_patience: int = 200
    compact_freeze_mode: str = "entropy"
    compact_freeze_min_interval: int = 5_000
    compact_freeze_probe_interval: int = 1_000
    compact_freeze_max_winner_change: float = 0.001
    compact_freeze_max_probe_drift: float = 0.02
    compact_freeze_max_taylor: float = 0.05
    compact_freeze_max_loss_increase: float = 0.01
    compact_freeze_probe_patience: int = 3
    compact_freeze_probe_batches: int = 4
    compact_freeze_loss_sigma: float = 2.0
    compact_dense_freeze: bool = False
    dense_freeze_fraction: float = 0.05
    dense_freeze_min_interval: int = 1_000
    dense_freeze_probe_interval: int = 1_000
    dense_freeze_probe_batches: int = 4
    dense_freeze_probe_patience: int = 3
    dense_freeze_max_loss_increase: float = 0.01
    dense_freeze_loss_sigma: float = 2.0

    @property
    def uses_compact_logic(self) -> bool:
        return self.model_variant == "build_model_cifar10_compact_small"

    @property
    def uses_classrouted_compact_head(self) -> bool:
        return self.model_variant in {
            "build_model_cifar10_small_compact_head",
            "build_model_cifar10_small_compact_head_wide75",
            "build_model_cifar10_small_classrouted_compact_head",
            "build_model_cifar10_compact_sixconv",
            "build_model_cifar10_small_mrlb_compact_head",
        }

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
        if self.uses_compact_logic:
            path = path / (
                f"compact_{self.compact_backend}_explore{self.compact_exploration_steps}_"
                f"resample{self.compact_resample_patience}_freeze{self.compact_freeze_patience}"
            )
        if self.uses_classrouted_compact_head:
            topology = (
                "compact_shared_head6_v2_plain_w80000x80000"
                if self.model_variant == "build_model_cifar10_compact_sixconv"
                else "compact_shared_head_v6_concat_w88000x88000"
            )
            if self.model_variant == "build_model_cifar10_small_mrlb_compact_head":
                topology = (
                    "compact_mrlb_v2_local64r1_cross64r2_"
                    f"route{self.mrlb_conv3_relational_channels}_w88000x88000"
                )
            if self.model_variant == "build_model_cifar10_small_compact_head":
                topology = "compact_shared_head2_v2_concat_w92400x92400"
            if self.model_variant == "build_model_cifar10_small_compact_head_wide75":
                topology = "compact_shared_head2_wide75_v2_concat_c224x224x896x896_w161700x161700"
            path = path / (
                f"{topology}_{self.compact_backend}_c16_"
                f"explore{self.compact_exploration_steps}_resample{self.compact_resample_patience}_"
                f"freeze{self.compact_freeze_patience}"
            )
        if self.compact_layer_aware_resampling:
            path = path / "layer_aware_resampling_v1_preferred25"
        if self.compact_efficient_layer_aware_resampling:
            path = path / "efficient_layer_aware_resampling_v2_conv25_dense1_u70_i18_n8_c4_dense2_u80_i12_n6_c2"
        if self.compact_aggressive_layer_aware_resampling:
            path = path / "aggressive_layer_aware_resampling_v1_conv25_d1_u50_i30_n5_c15_d2_u60_i25_n5_c10"
        if self.compact_freeze_mode == "none":
            path = path / "no_freeze_v1"
        if self.compact_freeze_mode == "hybrid":
            path = path / (
                f"hybrid_freeze_v3_probedriven_n{self.compact_freeze_probe_batches}_"
                f"sigma{self.compact_freeze_loss_sigma:g}_gap{self.compact_freeze_min_interval}_"
                f"probe{self.compact_freeze_probe_interval}_confirm{self.compact_freeze_probe_patience}_"
                f"drift{self.compact_freeze_max_probe_drift:g}_"
                f"flip{self.compact_freeze_max_winner_change:g}_"
                f"taylor{self.compact_freeze_max_taylor:g}_loss{self.compact_freeze_max_loss_increase:g}"
            )
        if self.compact_dense_freeze:
            path = path / (
                f"dense_groups_v1_frac{self.dense_freeze_fraction:g}_"
                f"gap{self.dense_freeze_min_interval}_probe{self.dense_freeze_probe_interval}_"
                f"n{self.dense_freeze_probe_batches}_confirm{self.dense_freeze_probe_patience}_"
                f"loss{self.dense_freeze_max_loss_increase:g}_sigma{self.dense_freeze_loss_sigma:g}"
            )
        if self.rewire_loss_probe:
            rewire_name = (
                "rewire_loss_probe_v4_all_layers_safe"
                if self.model_variant == "build_model_cifar10_smallmodel"
                else "rewire_loss_probe_v2"
            )
            path = path / (
                f"{rewire_name}_w{self.rewire_warmup}_i{self.rewire_interval}_"
                f"r{self.rewire_max_rows}_c{self.rewire_candidates}_"
                f"f{self.rewire_feature_pool}_e{self.rewire_entropy_threshold:g}_"
                f"d{self.rewire_entropy_delta:g}_m{self.rewire_min_improvement:g}"
            )
        return (
            path
            / self.parametrization
            / self.model_variant
            / self.experiment
            / f"seed_{self.seed}"
        )

    @property
    def input_encoding_slug(self) -> str:
        if self.input_encoding in {"fixed", "quantile", "thermometer"}:
            return f"{self.input_encoding}_{self.quantile_bits}bit"
        if self.input_encoding == "hybrid" and self.hybrid_features == "lbp":
            return "hybrid_q2_lbp2"
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
        if self.input_encoding == "thermometer":
            return self.num_input_bits
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
            if self.hybrid_features == "lbp":
                return 0
            return self.num_sine_bits or 2
        return 0

    @property
    def lbp_bits(self) -> int:
        return 2 if self.input_encoding == "hybrid" and self.hybrid_features == "lbp" else 0

    @property
    def total_input_bits(self) -> int:
        return self.quantile_bits + self.sine_bits + self.lbp_bits

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
        self._validate_rewiring()
        self._validate_compact()

    def _validate_compact(self) -> None:
        if self.compact_aggressive_layer_aware_resampling:
            if self.compact_layer_aware_resampling or self.compact_efficient_layer_aware_resampling:
                raise ValueError("Choose only one layer-aware resampling strategy.")
            if self.model_variant not in {
                "build_model_cifar10_compact_small",
                "build_model_cifar10_small_classrouted_compact_head",
                "build_model_cifar10_small_compact_head",
                "build_model_cifar10_small_compact_head_wide75",
            }:
                raise ValueError("--compact-aggressive-layer-aware-resampling requires a four-convolution, two-dense CompactLogic variant.")
        if self.compact_efficient_layer_aware_resampling:
            if self.compact_layer_aware_resampling:
                raise ValueError("Choose only one layer-aware resampling strategy.")
            if self.model_variant not in {
                "build_model_cifar10_compact_small",
                "build_model_cifar10_small_classrouted_compact_head",
                "build_model_cifar10_small_compact_head",
                "build_model_cifar10_small_compact_head_wide75",
            }:
                raise ValueError("--compact-efficient-layer-aware-resampling requires a four-convolution, two-dense CompactLogic variant.")
        if self.compact_layer_aware_resampling and self.model_variant not in {
            "build_model_cifar10_compact_small",
            "build_model_cifar10_small_classrouted_compact_head",
            "build_model_cifar10_small_compact_head",
            "build_model_cifar10_small_compact_head_wide75",
        }:
            raise ValueError("--compact-layer-aware-resampling requires a four-convolution, two-dense CompactLogic variant.")
        if self.model_variant == "build_model_cifar10_small_mrlb_compact_head":
            budget = self.mrlb_conv3_relational_channels
            if not isinstance(budget, int) or isinstance(budget, bool) or not 2 <= budget <= 510 or budget % 2:
                raise ValueError("mrlb_conv3_relational_channels must be even, from 2 to 510.")
        elif self.mrlb_conv3_relational_channels != 128:
            raise ValueError("--mrlb-conv3-relational-channels requires the MRLB model variant.")
        if self.compact_freeze_mode not in {"none", "entropy", "hybrid"}:
            raise ValueError("compact_freeze_mode must be none, entropy, or hybrid.")
        if self.compact_freeze_mode == "none" and self.compact_dense_freeze:
            raise ValueError("--compact-dense-freeze cannot be combined with --compact-freeze-mode none.")
        if not (self.uses_compact_logic or self.uses_classrouted_compact_head):
            if self.compact_dense_freeze:
                raise ValueError("Dense group freezing requires a CompactLogic model variant.")
            if self.compact_freeze_mode != "entropy":
                raise ValueError("Non-default compact freezing modes require a CompactLogic variant.")
            if self.parametrization == "compact" or self.input_encoding == "thermometer":
                raise ValueError("Compact parametrization and thermometer mode require a CompactLogic variant.")
            return
        if self.parametrization != "compact" or self.experiment != "pure_ce":
            raise ValueError("CompactLogic requires --parametrization compact --experiment pure_ce.")
        if self.start_temperature != 1.0:
            raise ValueError("CompactLogic uses unit-temperature candidate softmax.")
        if self.compact_backend not in {"auto", "cuda", "torch"}:
            raise ValueError("compact_backend must be auto, cuda, or torch.")
        if not 0 < self.compact_exploration_steps < self.iterations:
            raise ValueError("compact_exploration_steps must be positive and less than iterations.")
        if self.compact_resample_patience <= 0 or self.compact_freeze_patience <= 0:
            raise ValueError("CompactLogic patience values must be positive.")
        if self.compact_dense_freeze:
            if not math.isfinite(self.dense_freeze_fraction) or not 0 < self.dense_freeze_fraction <= 1:
                raise ValueError("dense_freeze_fraction must be finite and in (0,1].")
            if self.dense_freeze_min_interval < 0:
                raise ValueError("dense_freeze_min_interval must be nonnegative.")
            if self.dense_freeze_probe_interval <= 0 or self.dense_freeze_probe_patience <= 0:
                raise ValueError("Dense freeze probe interval and patience must be positive.")
            if self.dense_freeze_probe_batches < 2:
                raise ValueError("Dense freezing requires at least two probe batches.")
            for name in ("dense_freeze_max_loss_increase", "dense_freeze_loss_sigma"):
                if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                    raise ValueError(f"{name} must be finite and nonnegative.")
        if self.compact_freeze_mode == "hybrid":
            if self.compact_freeze_min_interval < 0:
                raise ValueError("compact_freeze_min_interval cannot be negative.")
            if self.compact_freeze_probe_interval <= 0 or self.compact_freeze_probe_patience <= 0:
                raise ValueError("Hybrid freeze probe interval and patience must be positive.")
            if self.compact_freeze_probe_batches <= 0:
                raise ValueError("compact_freeze_probe_batches must be positive.")
            for name in ("compact_freeze_max_winner_change", "compact_freeze_max_taylor",
                         "compact_freeze_max_loss_increase", "compact_freeze_loss_sigma",
                         "compact_freeze_max_probe_drift"):
                value = getattr(self, name)
                if not math.isfinite(value) or value < 0:
                    raise ValueError(f"{name} must be finite and nonnegative.")
            if self.compact_freeze_max_winner_change > 1:
                raise ValueError("compact_freeze_max_winner_change must be at most 1.")
            if self.compact_freeze_max_probe_drift > 1:
                raise ValueError("compact_freeze_max_probe_drift must be at most 1.")

    def _validate_rewiring(self) -> None:
        if not self.rewire_loss_probe:
            return
        if self.dataset != "cifar10" or self.model_variant not in {
            "build_model_cifar10_small_classrouted",
            "build_model_cifar10_smallmodel",
        }:
            raise ValueError(
                "Loss-probed rewiring currently supports only the baseline "
                "CIFAR-10 class-routed and small models."
            )
        if self.uses_phase_loss:
            raise ValueError("Loss-probed rewiring does not support Allen-Cahn phase loss.")
        if self.parametrization != "light":
            raise ValueError("Loss-probed rewiring requires LightLUT parametrization.")
        if not 0 <= self.rewire_warmup < self.iterations:
            raise ValueError("rewire_warmup must be in [0, iterations).")
        for name in (
            "rewire_interval", "rewire_max_rows", "rewire_candidates",
            "rewire_feature_pool",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive.")
        if self.rewire_feature_pool < 2:
            raise ValueError("rewire_feature_pool must be at least 2.")
        if not 0 <= self.rewire_entropy_threshold <= 1:
            raise ValueError("rewire_entropy_threshold must be in [0, 1].")
        if not 0 <= self.rewire_entropy_delta <= 1:
            raise ValueError("rewire_entropy_delta must be in [0, 1].")
        if self.rewire_min_improvement < 0:
            raise ValueError("rewire_min_improvement cannot be negative.")

    def _validate_input_encoding(self) -> None:
        if self.hybrid_features not in {"sine", "lbp"}:
            raise ValueError("hybrid_features must be 'sine' or 'lbp'.")
        if self.hybrid_features == "lbp":
            if self.dataset not in {"cifar10", "cifar100"} or self.input_encoding != "hybrid":
                raise ValueError("LBP features require CIFAR with input_encoding='hybrid'.")
            if self.quantile_bits != 2:
                raise ValueError("Q2+LBP encoding requires exactly 2 quantile bits.")
            if self.num_sine_bits is not None:
                raise ValueError("Remove --num-sine-bits when using --hybrid-features lbp.")
        if self.input_encoding not in INPUT_ENCODINGS:
            raise ValueError(
                f"Unknown input encoding {self.input_encoding!r}; "
                f"choose from {INPUT_ENCODINGS}."
            )
        bit_options = THERMOMETER_BIT_OPTIONS if self.input_encoding == "thermometer" else INPUT_BIT_OPTIONS
        if self.num_input_bits not in bit_options:
            raise ValueError(
                f"num_input_bits must be one of {bit_options}, "
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
            if self.hybrid_features == "sine" and self.sine_bits not in (1, 2, 4):
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
