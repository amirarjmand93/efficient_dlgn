"""Run the five CIFAR-10 infusion controls sequentially.

Run from the repository root with:
    python -m examples.phase_logic.ablation_control_expriments.infusion_control.run_all
"""

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

from examples.phase_logic.run_experiment import build_parser, config_from_args


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
ARMS = (
    ("original_baseline", "Original baseline", "Off", "Off", ("--compact-control-holdout",)),
    ("conv_only", "Conv-only", "On", "Off", ("--compact-progressive-aggressive-infusion", "--compact-infusion-scope", "conv")),
    ("dense_only", "Dense-only", "Off", "Levels 1–5", ("--compact-progressive-aggressive-infusion", "--compact-infusion-scope", "dense")),
    ("fixed_infusion", "Fixed infusion", "On", "Level 1 only", ("--compact-progressive-aggressive-infusion", "--compact-infusion-level-1-only")),
    ("full_progressive", "Full progressive infusion", "On", "Levels 1–5", ("--compact-progressive-aggressive-infusion",)),
)


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=HERE / "results")
    parser.add_argument("--seed", type=int, default=424244921)
    parser.add_argument("--iterations", type=int, default=1_500_000)
    parser.add_argument("--exploration-steps", type=int, default=800_000)
    parser.add_argument("--eval-frequency", type=int, default=15_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    parser.add_argument("--compact-backend", choices=("cuda", "torch", "auto"), default="cuda")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running training or writing files.")
    parser.add_argument("--skip-completed", action="store_true", help="Reuse completed runs with identical configurations.")
    return parser


def _training_args(args, name, policy_args):
    return [
        "--dataset", "cifar10",
        "--model-variant", "build_model_cifar10_small_compact",
        "--input-encoding", "hybrid", "--hybrid-features", "lbp",
        "--num-quantile-bits", "2", "--cifar-augmentation",
        "--experiment", "pure_ce", "--parametrization", "compact",
        "--compact-backend", args.compact_backend,
        "--compact-exploration-steps", str(args.exploration_steps),
        "--iterations", str(args.iterations),
        "--eval-frequency", str(args.eval_frequency),
        "--learning-rate", str(args.learning_rate),
        "--batch-size", str(args.batch_size),
        "--num-workers", str(args.num_workers),
        "--seed", str(args.seed), "--device", args.device,
        "--data-dir", str(REPO_ROOT / "data"),
        "--output-dir", str(args.output_root / "runs" / name),
        "--save-model", "--compact-freeze-mode", "hybrid",
        "--compact-freeze-max-taylor", "0.20",
        "--compact-freeze-max-loss-increase", "0.099",
        "--compact-freeze-max-probe-drift", "1.0",
        *policy_args,
    ]


def _write_reports(root, rows):
    fields = ("experiment", "conv_infusion", "dense_infusion", "test_accuracy_percent",
              "best_test_accuracy_percent", "run_dir")
    with (root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# CIFAR-10 infusion control",
        "",
        "Same seed, model, input encoding, 90% training split, augmentation,",
        "training budget, and hybrid freezing across all five runs.",
        "Accuracy is the final discrete test accuracy; best evaluated accuracy is in summary.csv.",
        "",
        "| Experiment | Conv1–3 infusion | Dense1–2 infusion | Test accuracy (%) |",
        "| --- | --- | --- | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['experiment']} | {row['conv_infusion']} | "
            f"{row['dense_infusion']} | {row['test_accuracy_percent']} |"
        )
    lines.append("")
    (root / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    args = _parser().parse_args()
    args.output_root = args.output_root.expanduser().resolve()
    prepared = []
    for name, label, conv, dense, policy_args in ARMS:
        train_args = _training_args(args, name, policy_args)
        config = config_from_args(build_parser().parse_args(train_args))
        config.validate()
        command = [sys.executable, "-u", "-m", "examples.phase_logic.run_experiment", *train_args]
        run_dir = config.run_dir
        summary_path = run_dir / "summary.json"
        completed = summary_path.is_file()
        if run_dir.exists() and not args.dry_run:
            if not (args.skip_completed and completed):
                raise FileExistsError(f"Run directory already exists: {run_dir}. Use --skip-completed to reuse finished runs.")
            recorded = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
            expected = config.to_dict()
            if any(recorded.get(key) != value for key, value in expected.items()):
                raise ValueError(f"Existing run has a different configuration: {run_dir}")
        prepared.append((name, label, conv, dense, command, run_dir, completed))

    if args.dry_run:
        for name, _, _, _, command, _, _ in prepared:
            print(f"{name}: {' '.join(command)}")
        return

    args.output_root.mkdir(parents=True, exist_ok=True)
    log_dir = args.output_root / "logs"
    log_dir.mkdir(exist_ok=True)
    rows = [
        {"experiment": label, "conv_infusion": conv, "dense_infusion": dense,
         "test_accuracy_percent": "TBD", "best_test_accuracy_percent": "TBD",
         "run_dir": str(run_dir)}
        for _, label, conv, dense, _, run_dir, _ in prepared
    ]
    _write_reports(args.output_root, rows)
    for index, (name, label, _, _, command, run_dir, completed) in enumerate(prepared):
        if not completed:
            print(f"\nStarting {index + 1}/{len(prepared)}: {label}", flush=True)
            with (log_dir / f"{name}.log").open("w", encoding="utf-8") as log:
                process = subprocess.Popen(
                    command, cwd=REPO_ROOT, stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, text=True, bufsize=1,
                )
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                return_code = process.wait()
            if return_code:
                raise RuntimeError(f"{label} failed with exit code {return_code}; see {log_dir / (name + '.log')}")
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        rows[index]["test_accuracy_percent"] = f"{100 * summary['final']['test_acc_discrete']:.2f}"
        rows[index]["best_test_accuracy_percent"] = f"{100 * summary['best_test_acc_discrete']:.2f}"
        _write_reports(args.output_root, rows)
        print(f"Completed {label}: {rows[index]['test_accuracy_percent']}% test accuracy", flush=True)


if __name__ == "__main__":
    main()
