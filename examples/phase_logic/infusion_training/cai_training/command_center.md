# CAI cross-dataset command center

Run from the repository root with the environment activated. This is a separate
copy of the working CAI trainer. It shares the controller, infusion policy,
metrics, and checkpoint format across four registered CompactLogic models.
The [existing CIFAR-10 α=0.95 run](legacy_runs.json) stays in its original folder.

The reference defaults to a local copy of that run's `config.json`.
The trainer reads its nested
`base_config`, then selects a dataset's model and preprocessing:

| Dataset | Model | Input |
| --- | --- | --- |
| CIFAR-10 | `build_model_cifar10_small_compact` | Q2 + local comparisons; crop/flip |
| CIFAR-100 | `build_model_cifar100_small_compact` | Q2 + local comparisons; crop/flip |
| MNIST | `build_model_mnist_small_compact` | 1-bit fixed input |
| Heartbeat | `build_model_heartbeat_small_compact` | Q2 + local comparisons |

All use the `strong_unary` policy, hybrid convolution freezing, 1,150,000
iterations, 600,000 exploration steps, evaluation every 50,000 steps,
`--step-down 0.05 --step-down-mode absolute`, and the same training settings
as the reference. No checkpoint weights are transferred: each run initializes
its own model. Validation uses a seeded 10% holdout from each training set.

## Overnight sweep

This one command runs the 12 dataset × initial-α experiments **sequentially**.
Completed run directories are skipped, so an interrupted sweep can be restarted.
An incomplete directory stops the sweep for inspection; no checkpoint is overwritten.
It does not schedule itself; start it when you are ready for overnight training.

```bash
python -m examples.phase_logic.infusion_training.cai_training.nightly
```

Inspect all 12 configurations first, without loading data or models:

```bash
python -m examples.phase_logic.infusion_training.cai_training.nightly --dry-run
```

The defaults are `--alphas 0.3 0.6 0.9` and `--seeds 424244921`. Add seeds
with, for example, `--seeds 424244921 4242449222`. You can also override
`--iterations`, `--exploration-steps`, or `--eval-frequency` for the whole sweep.
These are *initial* α values; feedback changes α during exploration.

## Ultra-unary overnight sweep

Run the same four datasets and seed as the `strong_unary` sweep, with initial
α of 0.6 and 0.9 only. These commands use 1,200,000 iterations, 600,000
exploration steps, evaluation every 50,000 steps, and all other settings from
the same reference config. First inspect the eight planned runs:

```bash
python -m examples.phase_logic.infusion_training.cai_training.nightly --infusion-profile ultra_unary --alphas 0.6 0.9 --seeds 424244921 --iterations 1200000 --exploration-steps 600000 --eval-frequency 50000 --dry-run
```

Then start the sequential overnight sweep:

```bash
python -m examples.phase_logic.infusion_training.cai_training.nightly --infusion-profile ultra_unary --alphas 0.6 0.9 --seeds 424244921 --iterations 1200000 --exploration-steps 600000 --eval-frequency 50000
```

Results go to separate
`cai_training/results/<dataset>/<model>/ultra_unary/alpha_<initial>/seed_424244921/`
directories. Completed runs are skipped; an incomplete existing run stops the
sweep for inspection. The `strong_unary` directories are untouched.

## Fixed-alpha overnight sweep

This control keeps candidate resampling enabled while disabling α feedback.
The three initial α values remain fixed at 0.3, 0.6, and 0.9. Results use
separate `strong_unary_fixed_alpha` directories and cannot overwrite adaptive
runs.

Inspect the planned runs:

```bash
python -m examples.phase_logic.infusion_training.cai_training.nightly --fixed-alpha --infusion-profile strong_unary --alphas 0.3 0.6 0.9 --seeds 424244921 --iterations 1200000 --exploration-steps 600000 --eval-frequency 50000 --dry-run
```

Start the sequential sweep:

```bash
python -m examples.phase_logic.infusion_training.cai_training.nightly --fixed-alpha --infusion-profile strong_unary --alphas 0.3 0.6 0.9 --seeds 424244921 --iterations 1200000 --exploration-steps 600000 --eval-frequency 50000
```

## Run one dataset

Change `--initial-alpha` to `0.3`, `0.6`, `0.9`, or another value in `[0,1]`.
The output path is automatically
`cai_training/results/<dataset>/<model>/strong_unary/alpha_<initial>/seed_<seed>/`.

```bash
python -m examples.phase_logic.infusion_training.cai_training.train --dataset cifar10 --initial-alpha 0.3 --seed 424244921
python -m examples.phase_logic.infusion_training.cai_training.train --dataset cifar100 --initial-alpha 0.3 --seed 424244921
python -m examples.phase_logic.infusion_training.cai_training.train --dataset mnist --initial-alpha 0.3 --seed 424244921
python -m examples.phase_logic.infusion_training.cai_training.train --dataset heartbeat --initial-alpha 0.3 --seed 424244921
```

The individual trainer fails if its destination already exists, protecting
completed checkpoints. Change `--seed` or `--output-dir` for another run.
`--reference-config` accepts a flat experiment config or a CAI config with
`base_config`. `--model-variant` can override the dataset's default model.
`--data-dir` points to the local dataset root. `--dry-run` prints the resolved
settings without opening datasets or models.

Each run writes `config.json`, `model_report.json`, `metrics.csv`,
`freeze_probes.csv`, `best_model.pt`, and `summary.json`; safe rollback snapshots
and gate dynamics are saved when applicable. Compare validation/test accuracy,
the α trajectory, gate changes, and resampling counts across matched seeds.
