"""Run the CAI dataset/initial-alpha ablation sequentially."""

import argparse
import subprocess
import sys

from .dataset_adapter import MODEL_BY_DATASET
from .train import build_parser, prepare_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=tuple(MODEL_BY_DATASET),
                        default=list(MODEL_BY_DATASET))
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.3, 0.6, 0.9])
    parser.add_argument("--seeds", nargs="+", type=int, default=[424244921])
    parser.add_argument("--infusion-profile", choices=("default", "strong_unary", "ultra_unary"),
                        default="strong_unary")
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--exploration-steps", type=int)
    parser.add_argument("--eval-frequency", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    for dataset in args.datasets:
        for alpha in args.alphas:
            for seed in args.seeds:
                command = [sys.executable, "-m",
                           "examples.phase_logic.infusion_training.cai_training.train",
                           "--dataset", dataset, "--seed", str(seed),
                           "--initial-alpha", str(alpha),
                           "--infusion-profile", args.infusion_profile]
                for name in ("iterations", "exploration_steps", "eval_frequency"):
                    value = getattr(args, name)
                    if value is not None:
                        command.extend(("--" + name.replace("_", "-"), str(value)))
                parsed = build_parser().parse_args(command[3:])
                _, _, output = prepare_config(parsed)
                if output.exists() and not args.dry_run:
                    if (output / "summary.json").is_file():
                        print(f"SKIP completed run: {output}", flush=True)
                        continue
                    raise FileExistsError(
                        f"Incomplete run directory exists: {output}. "
                        "Inspect it before starting a replacement run."
                    )
                print(f"RUN dataset={dataset} alpha={alpha:g} seed={seed} -> {output}",
                      flush=True)
                if args.dry_run:
                    command.append("--dry-run")
                subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
