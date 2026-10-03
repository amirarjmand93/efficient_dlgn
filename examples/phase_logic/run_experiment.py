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
        "--num-input-bits",
        type=int,
        choices=INPUT_BIT_OPTIONS,
        default=1,
        help="Legacy bit count for fixed/quantile modes.",
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
