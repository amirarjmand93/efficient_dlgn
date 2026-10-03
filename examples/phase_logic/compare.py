#!/usr/bin/env python3
"""Compare CE, annealing, and phase-loss training."""

import argparse
import csv
from dataclasses import replace
from pathlib import Path
from statistics import mean, stdev

from .config import (
    DATASETS,
    DEFAULT_MODEL_VARIANT,
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
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--dataset", choices=DATASETS, default="mnist")
    parser.add_argument(
        "--parametrization",
        choices=PARAMETRIZATIONS,
        default="light",
        help="Use the same LUT parametrization in all arms; phase requires light.",
    )
    parser.add_argument(
        "--model-variants",
        choices=MODEL_VARIANTS,
        nargs="+",
        default=None,
        help="Exact model builder function names from model.py.",
    )
    parser.add_argument(
        "--device", default="auto", choices=("auto", "cpu", "cuda", "mps")
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--input-encoding",
        choices=INPUT_ENCODINGS,
        default="fixed",
    )
    parser.add_argument(
        "--num-input-bits",
        type=int,
        choices=INPUT_BIT_OPTIONS,
        default=1,
    )
    parser.add_argument(
        "--num-quantile-bits",
        type=int,
        choices=INPUT_BIT_OPTIONS,
        default=None,
    )
    parser.add_argument(
        "--num-sine-bits",
        type=int,
        choices=INPUT_BIT_OPTIONS,
        default=None,
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
        "--output-dir", type=Path, default=Path("examples/phase_logic/results")
    )
    parser.add_argument("--iterations", type=int, default=10_300)
    parser.add_argument("--eval-frequency", type=int, default=100)
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
    parser.add_argument("--end-temperature", type=float, default=0.3)
    parser.add_argument("--temperature-warmup", type=int, default=200)
    parser.add_argument("--phase-margin", type=float, default=1.0)
    parser.add_argument("--phase-lambda-max", type=float, default=0.2)
    parser.add_argument("--phase-warmup", type=int, default=200)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.model_variants is None:
        args.model_variants = [DEFAULT_MODEL_VARIANT[args.dataset]]
    if args.parametrization != "light":
        parser.error("the annealing_phase comparison requires --parametrization light")
    base = ExperimentConfig(
        seed=args.seeds[0],
        dataset=args.dataset,
        parametrization=args.parametrization,
        model_variant=args.model_variants[0],
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
    )

    experiments = [
        "pure_ce",
        "annealing",
        "annealing_phase",
    ]

    rows = []
    for seed in args.seeds:
        for model_variant in args.model_variants:
            for experiment in experiments:
                config = replace(
                    base,
                    experiment=experiment,
                    model_variant=model_variant,
                    seed=seed,
                )
                summary = run_experiment(config)
                final = summary["final"]
                rows.append(
                    {
                        "experiment": experiment,
                        "dataset": args.dataset,
                        "model_variant": model_variant,
                        "parametrization": args.parametrization,
                        "input_encoding": args.input_encoding,
                        "cifar_augmentation": config.cifar_augmentation,
                        "num_input_bits": config.total_input_bits,
                        "num_quantile_bits": config.quantile_bits,
                        "num_sine_bits": config.sine_bits,
                        "sine_sigma": config.sine_sigma,
                        "sine_seed": config.sine_seed,
                        "seed": seed,
                        "parameter_count": summary["parameter_count"],
                        "lr_schedule": config.lr_schedule,
                        "end_learning_rate": config.end_learning_rate,
                        "best_test_acc_discrete": summary[
                            "best_test_acc_discrete"
                        ],
                        "final_train_loss_relaxed": final[
                            "train_loss_relaxed"
                        ],
                        "final_train_acc_relaxed": final[
                            "train_acc_relaxed"
                        ],
                        "final_test_acc_discrete": final[
                            "test_acc_discrete"
                        ],
                        "final_test_acc_relaxed": final[
                            "test_acc_relaxed"
                        ],
                        "final_accuracy_gap": final["accuracy_gap"],
                        "final_phase_energy": final[
                            "current_phase_energy"
                        ],
                        "final_fraction_outside_margin": final[
                            "fraction_outside_margin"
                        ],
                    }
                )

    comparison_dir = args.output_dir / args.dataset
    if args.dataset in {"cifar10", "cifar100"}:
        comparison_dir = comparison_dir / base.input_encoding_slug
        if base.cifar_augmentation:
            comparison_dir = comparison_dir / "crop_flip_augmentation"
    if base.lr_schedule == "cosine":
        comparison_dir = comparison_dir / base.lr_schedule_slug
    comparison_dir = comparison_dir / args.parametrization
    comparison_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = comparison_dir / "comparison.csv"
    with comparison_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    paired_rows = []
    for seed in args.seeds:
        for model_variant in args.model_variants:
            by_experiment = {
                row["experiment"]: row
                for row in rows
                if row["seed"] == seed
                and row["model_variant"] == model_variant
            }
            annealing = by_experiment["annealing"]
            treatment = by_experiment["annealing_phase"]
            paired_rows.append(
                {
                    "method": "annealing_phase",
                    "dataset": args.dataset,
                    "model_variant": model_variant,
                    "seed": seed,
                    "parametrization": args.parametrization,
                    "discrete_accuracy_delta": (
                        treatment["final_test_acc_discrete"]
                        - annealing["final_test_acc_discrete"]
                    ),
                    "relaxed_accuracy_delta": (
                        treatment["final_test_acc_relaxed"]
                        - annealing["final_test_acc_relaxed"]
                    ),
                    "accuracy_gap_delta": (
                        treatment["final_accuracy_gap"]
                        - annealing["final_accuracy_gap"]
                    ),
                }
            )

    head_rows = []
    head_models = {
        "mnist": ("build_model", "build_model_linear_head"),
        "cifar10": (
            "build_model_cifar10_org",
            "build_model_cifar10_linear_head",
        ),
    }.get(args.dataset)
    if head_models is not None and set(head_models).issubset(args.model_variants):
        original_name, linear_name = head_models
        for seed in args.seeds:
            for experiment in experiments:
                by_model = {
                    row["model_variant"]: row
                    for row in rows
                    if row["seed"] == seed
                    and row["experiment"] == experiment
                }
                original = by_model[original_name]
                linear = by_model[linear_name]
                head_rows.append(
                    {
                        "experiment": experiment,
                        "dataset": args.dataset,
                        "parametrization": args.parametrization,
                        "seed": seed,
                        "original_parameter_count": original[
                            "parameter_count"
                        ],
                        "linear_parameter_count": linear["parameter_count"],
                        "parameter_count_delta": (
                            linear["parameter_count"]
                            - original["parameter_count"]
                        ),
                        "train_acc_relaxed_delta": (
                            linear["final_train_acc_relaxed"]
                            - original["final_train_acc_relaxed"]
                        ),
                        "test_acc_relaxed_delta": (
                            linear["final_test_acc_relaxed"]
                            - original["final_test_acc_relaxed"]
                        ),
                        "test_acc_discrete_delta": (
                            linear["final_test_acc_discrete"]
                            - original["final_test_acc_discrete"]
                        ),
                    }
                )

    paired_path = comparison_dir / "paired_differences.csv"
    with paired_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired_rows[0]))
        writer.writeheader()
        writer.writerows(paired_rows)

    head_path = comparison_dir / "head_differences.csv"
    if head_rows:
        with head_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(head_rows[0]))
            writer.writeheader()
            writer.writerows(head_rows)

    print(f"\nComparison ({args.dataset}/{args.parametrization})")
    for row in rows:
        print(
            f"  {row['experiment']:24s} "
            f"model={row['model_variant']:18s} seed={row['seed']:4d} "
            f"parameters={row['parameter_count']:6d} "
            f"train_loss_relaxed={row['final_train_loss_relaxed']:.4f} "
            f"train_acc_relaxed={row['final_train_acc_relaxed']:.4f} "
            f"test_acc_discrete={row['final_test_acc_discrete']:.4f} "
            f"test_acc_relaxed={row['final_test_acc_relaxed']:.4f} "
            f"gap={row['final_accuracy_gap']:+.4f}"
        )
    for model_variant in args.model_variants:
        method_rows = [
            row
            for row in paired_rows
            if row["model_variant"] == model_variant
        ]
        print(
            "\nPaired effect of annealing_phase relative to annealing "
            f"({model_variant})"
        )
        for metric in (
            "discrete_accuracy_delta",
            "relaxed_accuracy_delta",
            "accuracy_gap_delta",
        ):
            values = [row[metric] for row in method_rows]
            sample_std = stdev(values) if len(values) > 1 else 0.0
            print(
                f"  {metric}={mean(values):+.4f} "
                f"(sample std={sample_std:.4f})"
            )
    if head_rows:
        print("\nHead effect (linear_head - original_group_sum)")
        for row in head_rows:
            print(
                f"  {row['experiment']:24s} seed={row['seed']:4d} "
                f"parameters={row['linear_parameter_count']} vs "
                f"{row['original_parameter_count']} "
                f"train_acc_relaxed={row['train_acc_relaxed_delta']:+.4f} "
                f"test_acc_relaxed={row['test_acc_relaxed_delta']:+.4f} "
                f"test_acc_discrete={row['test_acc_discrete_delta']:+.4f}"
            )
    print(f"Saved comparison to {comparison_path}")
    print(f"Saved paired differences to {paired_path}")
    if head_rows:
        print(f"Saved head differences to {head_path}")


if __name__ == "__main__":
    main()
