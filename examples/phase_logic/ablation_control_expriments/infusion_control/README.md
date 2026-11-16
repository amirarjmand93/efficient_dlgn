# CIFAR-10 infusion control experiment

From the repository root, run all five experiments sequentially:

```bash
python -m examples.phase_logic.ablation_control_expriments.infusion_control.run_all
```

The five arms are original baseline, Conv1–3 only, Dense1–2 only, fixed
Level 1 on both, and full progressive infusion. Every arm uses
`build_model_cifar10_small_compact`, Q2+LBP2, crop and flip augmentation,
the same seed and 90% training split, 1.5 million steps, 800,000 exploration
steps, and hybrid freezing. Only the candidate infusion policy changes.

Outputs are stored under `infusion_control/results`: individual training
runs in `runs/`, logs in `logs/`, and the comparison in `summary.md` and
`summary.csv`. The table is updated after each completed run. Use
`--skip-completed` to continue after an interruption, or `--dry-run` to print
the five commands without starting training. `--seed`, `--iterations`,
`--exploration-steps`, `--eval-frequency`, `--batch-size`, `--num-workers`,
`--learning-rate`, `--device`, `--compact-backend`, and `--output-root` are
configurable; they apply to all five arms together.

The reported test accuracy is from the final discrete model at the last
evaluation. One seed provides a controlled comparison, not an estimate of
run-to-run variation.
