#!/usr/bin/env python3
"""Run one pure-CE or phase-logic experiment."""

import argparse
from pathlib import Path

from .config import (
    DATASETS,
    DEFAULT_MODEL_VARIANT,
    EXPERIMENTS,
    INPUT_BIT_OPTIONS,
    INPUT_ENCODINGS,
    LR_SCHEDULES,
    MODEL_VARIANTS,
    PARAMETRIZATIONS,
    THERMOMETER_BIT_OPTIONS,
    ExperimentConfig,
)
from .training import run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=EXPERIMENTS, default="annealing")
    parser.add_argument("--dataset", choices=DATASETS, default="mnist")
    parser.add_argument(
        "--parametrization",
        choices=PARAMETRIZATIONS,
        default="light",
        help="LUT representation used by every logic layer.",
    )
    parser.add_argument(
        "--model-variant",
        choices=MODEL_VARIANTS,
        default=None,
        help="Exact model builder function name from model.py.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", default="auto", choices=("auto", "cpu", "cuda", "mps")
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--input-encoding",
        choices=INPUT_ENCODINGS,
        default="fixed",
        help="CIFAR input encoding mode.",
    )
    parser.add_argument(
        "--hybrid-features",
        choices=("sine", "lbp"),
        default="sine",
        help="Hybrid feature branch: existing sine bits, or Q2 + two local comparison bits.",
    )
    parser.add_argument(
        "--num-input-bits",
        type=int,
        choices=THERMOMETER_BIT_OPTIONS,
        default=1,
        help="Bit count for fixed/quantile modes, or uniform thermometer thresholds.",
    )
    parser.add_argument(
        "--num-quantile-bits",
        type=int,
        choices=INPUT_BIT_OPTIONS,
        default=None,
        help="Quantile bits per RGB channel; overrides --num-input-bits.",
    )
    parser.add_argument(
        "--num-sine-bits",
        type=int,
        choices=INPUT_BIT_OPTIONS,
        default=None,
        help="Sinusoidal bits per RGB channel.",
    )
    parser.add_argument("--sine-sigma", type=float, default=1.0)
    parser.add_argument("--sine-seed", type=int, default=0)
    parser.add_argument(
        "--cifar-augmentation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Apply a reflection-padded random crop and random horizontal flip "
            "to CIFAR training batches (default: off)."
        ),
    )
    parser.add_argument(
        "--cifar-random-erasing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Add RandomErasing after CIFAR crop/flip augmentation "
            "(default: off; requires --cifar-augmentation)."
        ),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("examples/phase_logic/results")
    )
    parser.add_argument("--iterations", type=int, default=1_000)
    parser.add_argument("--eval-frequency", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument(
        "--lr-schedule",
        choices=LR_SCHEDULES,
        default="constant",
        help="Learning-rate schedule (default: constant).",
    )
    parser.add_argument(
        "--end-learning-rate",
        type=float,
        default=0.002,
        help="Final learning rate for cosine decay (default: 0.002).",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--start-temperature", type=float, default=1.0)
    parser.add_argument("--end-temperature", type=float, default=0.2)
    parser.add_argument("--temperature-warmup", type=int, default=200)
    parser.add_argument("--phase-margin", type=float, default=1.0)
    parser.add_argument("--phase-lambda-max", type=float, default=0.1)
    parser.add_argument("--phase-warmup", type=int, default=200)
    parser.add_argument("--save-model", action="store_true")
    parser.add_argument(
        "--rewire-loss-probe", action="store_true",
        help="Opt in to loss-probed rewiring (all small-model logic layers, or the first class-routed dense layer).",
    )
    parser.add_argument("--rewire-warmup", type=int, default=10_000)
    parser.add_argument("--rewire-interval", type=int, default=1_000)
    parser.add_argument("--rewire-max-rows", type=int, default=8)
    parser.add_argument("--rewire-candidates", type=int, default=8)
    parser.add_argument("--rewire-feature-pool", type=int, default=256)
    parser.add_argument("--rewire-entropy-threshold", type=float, default=0.15)
    parser.add_argument("--rewire-entropy-delta", type=float, default=0.002)
    parser.add_argument("--rewire-min-improvement", type=float, default=0.0)
    parser.add_argument("--compact-backend", choices=("auto", "cuda", "torch"), default="auto")
    parser.add_argument("--mrlb-conv3-relational-channels", type=int, default=128,
                        help="MRLB-only: even Conv3 output budget for relational routes (2..510), split equally across branches.")
    parser.add_argument("--compact-exploration-steps", type=int, default=400_000)
    parser.add_argument("--compact-resample-patience", type=int, default=100)
    parser.add_argument("--compact-layer-aware-resampling", action="store_true",
                        help="Opt in to 25%% preferred / 75%% uniform replacement opcodes: AND/OR in Conv1-3, XOR/XNOR in Dense2 (four-conv models only).")
    parser.add_argument("--compact-efficient-layer-aware-resampling", action="store_true",
                        help="Opt in to ELAR: Conv1-3 75%% uniform + 25%% AND/OR; Conv4 uniform; Dense1 uniform/identity/NOT/constants=70/18/8/4%%; Dense2=80/12/6/2%%. Do not combine with the original layer-aware flag.")
    parser.add_argument("--compact-aggressive-layer-aware-resampling", action="store_true",
                        help="Separate aggressive policy: Conv1-3 75%% uniform + 25%% AND/OR; Conv4 uniform; Dense1 uniform/identity/NOT/constants=50/30/5/15%%; Dense2=60/25/5/10%%. Do not combine with other layer-aware flags.")
    parser.add_argument("--compact-freeze-patience", type=int, default=200)
    parser.add_argument("--compact-freeze-mode", choices=("none", "entropy", "hybrid"), default="entropy",
                        help="Disable freezing (none), original entropy freezing, or probe-driven hybrid freezing. Exploration resampling is unchanged.")
    parser.add_argument("--compact-freeze-min-interval", type=int, default=5_000)
    parser.add_argument("--compact-freeze-probe-interval", type=int, default=1_000)
    parser.add_argument("--compact-freeze-max-winner-change", type=float, default=0.001,
                        help="Winner-change threshold for hybrid stability diagnostics only.")
    parser.add_argument("--compact-freeze-max-probe-drift", type=float, default=0.02,
                        help="Maximum changed-winner fraction relative to a confirmation streak's anchor.")
    parser.add_argument("--compact-freeze-max-taylor", type=float, default=0.05)
    parser.add_argument("--compact-freeze-max-loss-increase", type=float, default=0.01)
    parser.add_argument("--compact-freeze-probe-patience", type=int, default=3)
    parser.add_argument("--compact-freeze-probe-batches", type=int, default=4,
                        help="Training batches checked at each fixed-snapshot hybrid probe.")
    parser.add_argument("--compact-freeze-loss-sigma", type=float, default=2.0,
                        help="Standard-error multiplier for the paired hardening-loss guard.")
    parser.add_argument("--compact-dense-freeze", action="store_true",
                        help="Opt in to loss-probed dense group freezing, after all convs freeze.")
    parser.add_argument("--dense-freeze-fraction", type=float, default=0.05)
    parser.add_argument("--dense-freeze-min-interval", type=int, default=1_000,
                        help="Recovery steps after conv completion and each committed dense group.")
    parser.add_argument("--dense-freeze-probe-interval", type=int, default=1_000)
    parser.add_argument("--dense-freeze-probe-batches", type=int, default=4)
    parser.add_argument("--dense-freeze-probe-patience", type=int, default=3)
    parser.add_argument("--dense-freeze-max-loss-increase", type=float, default=0.01)
    parser.add_argument("--dense-freeze-loss-sigma", type=float, default=2.0)
    return parser


def config_from_args(args: argparse.Namespace, experiment: str | None = None):
    return ExperimentConfig(
        experiment=experiment or args.experiment,
        dataset=args.dataset,
        parametrization=args.parametrization,
        model_variant=(
            args.model_variant or DEFAULT_MODEL_VARIANT[args.dataset]
        ),
        seed=args.seed,
        device=args.device,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        input_encoding=args.input_encoding,
        hybrid_features=args.hybrid_features,
        num_input_bits=args.num_input_bits,
        num_quantile_bits=args.num_quantile_bits,
        num_sine_bits=args.num_sine_bits,
        sine_sigma=args.sine_sigma,
        sine_seed=args.sine_seed,
        cifar_augmentation=args.cifar_augmentation,
        cifar_random_erasing=args.cifar_random_erasing,
        iterations=args.iterations,
        eval_frequency=args.eval_frequency,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lr_schedule=args.lr_schedule,
        end_learning_rate=args.end_learning_rate,
        num_workers=args.num_workers,
        start_temperature=args.start_temperature,
        end_temperature=args.end_temperature,
        temperature_warmup=args.temperature_warmup,
        phase_margin=args.phase_margin,
        phase_lambda_max=args.phase_lambda_max,
        phase_warmup=args.phase_warmup,
        save_model=args.save_model,
        rewire_loss_probe=args.rewire_loss_probe,
        rewire_warmup=args.rewire_warmup,
        rewire_interval=args.rewire_interval,
        rewire_max_rows=args.rewire_max_rows,
        rewire_candidates=args.rewire_candidates,
        rewire_feature_pool=args.rewire_feature_pool,
        rewire_entropy_threshold=args.rewire_entropy_threshold,
        rewire_entropy_delta=args.rewire_entropy_delta,
        rewire_min_improvement=args.rewire_min_improvement,
        compact_backend=args.compact_backend,
        mrlb_conv3_relational_channels=args.mrlb_conv3_relational_channels,
        compact_exploration_steps=args.compact_exploration_steps,
        compact_resample_patience=args.compact_resample_patience,
        compact_layer_aware_resampling=args.compact_layer_aware_resampling,
        compact_efficient_layer_aware_resampling=args.compact_efficient_layer_aware_resampling,
        compact_aggressive_layer_aware_resampling=args.compact_aggressive_layer_aware_resampling,
        compact_freeze_patience=args.compact_freeze_patience,
        compact_freeze_mode=args.compact_freeze_mode,
        compact_freeze_min_interval=args.compact_freeze_min_interval,
        compact_freeze_probe_interval=args.compact_freeze_probe_interval,
        compact_freeze_max_winner_change=args.compact_freeze_max_winner_change,
        compact_freeze_max_probe_drift=args.compact_freeze_max_probe_drift,
        compact_freeze_max_taylor=args.compact_freeze_max_taylor,
        compact_freeze_max_loss_increase=args.compact_freeze_max_loss_increase,
        compact_freeze_probe_patience=args.compact_freeze_probe_patience,
        compact_freeze_probe_batches=args.compact_freeze_probe_batches,
        compact_freeze_loss_sigma=args.compact_freeze_loss_sigma,
        compact_dense_freeze=args.compact_dense_freeze,
        dense_freeze_fraction=args.dense_freeze_fraction,
        dense_freeze_min_interval=args.dense_freeze_min_interval,
        dense_freeze_probe_interval=args.dense_freeze_probe_interval,
        dense_freeze_probe_batches=args.dense_freeze_probe_batches,
        dense_freeze_probe_patience=args.dense_freeze_probe_patience,
        dense_freeze_max_loss_increase=args.dense_freeze_max_loss_increase,
        dense_freeze_loss_sigma=args.dense_freeze_loss_sigma,
    )


def main() -> None:
    args = build_parser().parse_args()
    summary = run_experiment(config_from_args(args))
    final = summary["final"]
    print(
        f"Finished {summary['dataset']}/{summary['experiment']}/"
        f"{summary['model_variant']}: "
        f"parameters={summary['parameter_count']}; "
        f"final discrete accuracy={final['test_acc_discrete']:.4f}; "
        f"results saved under {config_from_args(args).run_dir}"
    )


if __name__ == "__main__":
    main()
