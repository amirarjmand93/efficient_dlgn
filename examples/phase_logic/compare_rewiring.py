"""Compare matched fixed and loss-probed CIFAR-10 training runs."""

import argparse
import json
from pathlib import Path
from statistics import mean


MATCHED_SETTINGS = (
    "dataset", "model_variant", "experiment", "parametrization", "device",
    "data_dir", "num_workers",
    "input_encoding", "num_input_bits", "num_quantile_bits", "num_sine_bits",
    "sine_sigma", "sine_seed", "cifar_augmentation", "cifar_random_erasing",
    "iterations", "eval_frequency", "batch_size", "learning_rate",
    "lr_schedule", "end_learning_rate", "start_temperature",
    "end_temperature", "temperature_warmup", "phase_margin",
    "phase_lambda_max", "phase_warmup",
)


def _read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def compare_runs(baseline_root: Path, rewired_root: Path, seeds: list[int]) -> list[dict]:
    rows = []
    for seed in seeds:
        fixed_dir = baseline_root / f"seed_{seed}"
        rewired_dir = rewired_root / f"seed_{seed}"
        fixed_config = _read_json(fixed_dir / "config.json")
        rewired_config = _read_json(rewired_dir / "config.json")
        mismatches = [
            name for name in MATCHED_SETTINGS
            if fixed_config.get(name) != rewired_config.get(name)
        ]
        if fixed_config.get("seed") != seed or rewired_config.get("seed") != seed:
            mismatches.append("seed")
        if fixed_config.get("rewire_loss_probe") or not rewired_config.get("rewire_loss_probe"):
            mismatches.append("rewire_loss_probe")
        if fixed_config.get("resolved_device") != rewired_config.get("resolved_device"):
            mismatches.append("resolved_device")
        if mismatches:
            raise ValueError(f"seed {seed} has unmatched settings: {mismatches}")

        fixed = _read_json(fixed_dir / "summary.json")
        rewired = _read_json(rewired_dir / "summary.json")
        rows.append({
            "seed": seed,
            "fixed_discrete": fixed["final"]["test_acc_discrete"],
            "rewired_discrete": rewired["final"]["test_acc_discrete"],
            "fixed_relaxed": fixed["final"]["test_acc_relaxed"],
            "rewired_relaxed": rewired["final"]["test_acc_relaxed"],
            "fixed_seconds": fixed.get("training_elapsed_seconds"),
            "rewired_seconds": rewired.get("training_elapsed_seconds"),
            "probe_seconds": rewired.get("rewiring", {}).get("elapsed_seconds"),
            "accepted_rows": rewired.get("rewiring", {}).get("accepted_rows"),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--rewired-root", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    args = parser.parse_args()
    rows = compare_runs(args.baseline_root, args.rewired_root, args.seeds)

    def seconds(value):
        return "-" if value is None else f"{value:.1f}"

    print("seed  fixed_discrete  rewired_discrete  delta  fixed_relaxed  rewired_relaxed  fixed_s  rewired_s  probe_s  accepted")
    for row in rows:
        delta = row["rewired_discrete"] - row["fixed_discrete"]
        print(
            f"{row['seed']:4d}  {row['fixed_discrete']:.4f}          "
            f"{row['rewired_discrete']:.4f}            {delta:+.4f}  "
            f"{row['fixed_relaxed']:.4f}         {row['rewired_relaxed']:.4f}           "
            f"{seconds(row['fixed_seconds'])}  "
            f"{seconds(row['rewired_seconds'])}  "
            f"{seconds(row['probe_seconds'])}  {row['accepted_rows']}"
        )
    print(
        "mean discrete delta: "
        f"{mean(row['rewired_discrete'] - row['fixed_discrete'] for row in rows):+.4f}"
    )
    timed = [
        row for row in rows
        if row["fixed_seconds"] is not None
        and row["rewired_seconds"] is not None
        and row["fixed_seconds"] > 0
    ]
    if timed:
        print(
            "mean training runtime ratio (rewired/fixed): "
            f"{mean(row['rewired_seconds'] / row['fixed_seconds'] for row in timed):.3f}x"
        )


if __name__ == "__main__":
    main()
