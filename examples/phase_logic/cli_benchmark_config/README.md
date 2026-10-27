# Benchmark experiment plan

Datasets: **CIFAR-10, CIFAR-100, MNIST**. Sizes: **small, medium, large**.
Here, **freezing means hybrid convolution freezing**, not entropy or dense-group
freezing. No training is launched by this document.

## Required experiments

| Experiment | ELAR | Hybrid freezing | Coverage | Runs per seed |
|---|---|---|---|---:|
| Baseline | Off (uniform resampling) | Off | 3 datasets × 3 sizes | 9 |
| Full LightoLogic | On | On | Same 9 dataset/size combinations | 9 |
| Freezing-only | Off | On | Medium size on each dataset | 3 |
| ELAR-only | On | Off | Medium size on each dataset | 3 |
| **Total** | | | | **24** |

“Full LightoLogic” means ELAR + hybrid freezing on the **same architecture and
encoder** as its baseline; other architecture/encoder innovations are not
isolated by these tests. Use the same three seeds for every experiment:
**4242449, 42424490, 424244999**. Final total: **72 runs**.

## Controls and prerequisites

- Match architecture, widths, encoder, augmentation, optimizer/LR schedule,
  batch size, training steps, and evaluation frequency within each comparison.
  Keep the same exploration cutoff and resampling triggers; vary only the two
  studied mechanisms. Disable dense-group freezing in all runs.
- Map small/medium/large to explicit model variants before starting. Preserve
  existing models; compatible CIFAR-100/MNIST variants and method support are
  still required. Confirm ELAR's layer assignments for each new architecture.
- Use `--compact-freeze-mode none` for freezing-off and
  `--compact-freeze-mode hybrid` for freezing-on. Omitting freezing flags
  enables default entropy freezing and is **not** a valid off condition.
- Use a held-out training split for tuning and checkpoint selection. Reserve
  the test set for final evaluation; fit encoder thresholds on training only.
- Validate each CLI and give each run a separate result directory. Record
  failures and unsupported configurations; do not silently substitute models.

## Results

One row per seed/run; aggregate mean ± standard deviation across the three
seeds. Primary metric: **discrete test accuracy** at the validation-selected
checkpoint. Also record relaxed accuracy/gap, training wall time, training
parameter count, gate evaluations/image, and simplified IR gate count. State
whether encoder/GroupSum costs are included; IR gate counts are not FPGA LUTs.

| Dataset | Size / model variant | Experiment | Seed | Status | Discrete acc. | Relaxed acc. / gap | Params / gate counts | Time | Results path / notes |
|---|---|---|---|---|---|---|---|---|---|
| — | — | — | — | Planned | — | — | — | — | — |

Optional hardware comparison: synthesize selected checkpoints with identical
tool settings on the same FPGA parts. Report LUT/FF use, critical-path logic
levels, and estimated delay separately from accuracy; label synthesis versus
post-route results and encoder exclusions explicitly.
