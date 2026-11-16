# Progressive Aggressive Infusion

This package provides the reusable `ProgressiveAggressiveInfusion` controller,
the five fixed candidate mixtures, and checkpoint save/load helpers. It does not
replace or alter the existing aggressive layer-aware mode.

The training integration holds out a deterministic, unaugmented 10% of the
training set. Every normal evaluation interval it measures discrete validation
accuracy and cross-entropy. Three stable, acceptable evaluations advance one
level. Before advancing, it saves `progressive_infusion_safe.pt`; a significant
regression restores that model, optimizer, CompactLogic controller, and infusion
state. Transitions are appended to `progressive_infusion_transitions.csv`.
After the first increase, accuracy is protected relative to the best safe
checkpoint rather than an absolute floor. The default guards tolerate a 0.3
percentage-point accuracy change and 0.05 loss increase while establishing
stability, require 15,000 training steps between level increases, and roll back
on a 1.5 percentage-point accuracy drop or 0.10 loss increase relative to the
last safe point. Rollback is immediate and is not delayed by the cooldown.

| Level | Uniform | Identity A/B | NOT A/B | TRUE/FALSE |
| --- | ---: | ---: | ---: | ---: |
| 1 — Aggressive | 50% | 30% | 5% | 15% |
| 2 — Strong | 40% | 25% | 10% | 25% |
| 3 — Very Strong | 30% | 20% | 10% | 40% |
| 4 — Extreme | 25% | 15% | 10% | 50% |
| 5 — Maximum | 15% | 10% | 5% | 70% |

Enable it by adding:

```bash
--compact-progressive-aggressive-infusion
```

## Unary-Biased schedule

Use the separate opt-in flag below to keep constants at 15% and increase
identity and inversion preferences across five levels. The original flag
above retains its existing percentages and results path.

```bash
--compact-progressive-aggressive-infusion-unary-biased
```

| Level | Uniform | Identity A/B | NOT A/B | TRUE/FALSE |
| --- | ---: | ---: | ---: | ---: |
| 1 | 50% | 30% | 5% | 15% |
| 2 | 45% | 30% | 10% | 15% |
| 3 | 40% | 35% | 10% | 15% |
| 4 | 35% | 35% | 15% | 15% |
| 5 | 30% | 40% | 15% | 15% |

The Conv1–3 policy and validation-driven level changes are the same as in
the original infusion. Unary-Biased runs use a `_unary_biased` results suffix.
Resume them with the Unary-Biased flag and the original scope and start level.

To begin at Level 2 or 3, add `--compact-infusion-start-level 2` or
`--compact-infusion-start-level 3`. The selected mixture applies from the
first training step; the usual validation checks still control later increases
and rollback. Valid values are 1 through 5, and the default remains 1.
Runs starting above Level 1 get a `_start2`, `_start3`, etc. results directory.
When resuming such a run, pass its original start level together with the
resume checkpoint; the saved current level is restored from that checkpoint.

Three opt-in ablations keep the default `all` scope and adaptive progression
unchanged:

```bash
--compact-progressive-aggressive-infusion --compact-infusion-scope conv
--compact-progressive-aggressive-infusion --compact-infusion-scope dense
--compact-progressive-aggressive-infusion --compact-infusion-level-1-only
```

`conv` applies the 25% AND/OR prior to Conv1–3 and uses uniform resampling
in Conv4 and both dense layers. The convolution prior is identical at every
infusion level, so this mode stays at Level 1. `dense` leaves every convolution
uniform and advances the dense candidate mixtures through the five levels.
`--compact-infusion-level-1-only` holds either scope at Level 1; without a
scope flag, it applies Level 1 to both convolution and dense layers. Each
option gets a distinct results directory. Resume with the same scope and
Level-1-only setting used when the checkpoint was saved.

Resume from a safe or progressive best checkpoint with:

```bash
--compact-progressive-aggressive-infusion \
--compact-infusion-resume-checkpoint path/to/progressive_infusion_safe.pt
```

For another Python training loop, import `ProgressiveAggressiveInfusion`,
`save_infusion_checkpoint`, and `load_infusion_checkpoint` from
`examples.phase_logic.infusion_training`. Call `observe()` with discrete
validation metrics; on an `advance` decision, save first and then call
`advance()`. On `rollback`, load the safe checkpoint and call
`reset_after_rollback()`.



# DLGN Hardware Comparison Best model: 92,400 Neurons With vs. Without Infusion

Both configurations have 92,400 neurons in each dense layer (184,800 dense neurons total) and were evaluated on the Xilinx Zynq-7035 (XC7Z035). The Boolean-function counts refer to the two dense layers combined; FPGA resources and timing are from post-route Vivado reports.

| Metric | No Infusion | With Infusion |
|---|---:|---:|
| Dense neurons (total) | 184,800 | 184,800 |
| Simple Boolean functions | 85,089 | **98,732** |
| Nontrivial Boolean functions | 99,711 | **86,068** |
| XOR + XNOR | 47,706 | **38,636** |
| Physical LUTs | 694 | **673** |
| Flip-flops | 472 | **468** |
| BRAM36 equivalents | 306.5 | **259.5** |
| BRAM utilization | 61.3% | **51.9%** |
| Setup slack (WNS) | +1.583 ns | **+2.079 ns** |
| Effective setup timing (10 ns − WNS) | 8.417 ns | **7.921 ns** |
| Hold slack (WHS) | +0.123 ns | +0.126 ns |
| Routed nets | 1,966 | 1,903 |
| Routing errors | 0 | 0 |
| Estimated on-chip power (Vivado GUI) | 11.279 W | **8.491 W** |
| Power-estimation confidence | Low | Low |

## Short explanation

At the same nominal dense-layer width, **Infusion increases simple Boolean functions and reduces nontrivial functions by 13.7%** (99,711 to 86,068). The post-route implementation also uses **15.3% fewer BRAM36-equivalent blocks** (306.5 to 259.5), **21 fewer LUTs**, and has **0.496 ns more setup slack** at 100 MHz. Both designs routed successfully and met reported internal setup and hold timing.

**Important limitations:** The 8.417 ns and 7.921 ns figures are effective setup timing values calculated from clock period minus WNS, **not total inference latency**. The power values are **low-confidence Vivado estimates**, not measured consumption; the apparent power decrease should not be attributed to Infusion without comparable switching-activity (SAIF) data. Equal accuracy and fully matched training conditions have not been established by these hardware reports.

best model: examples/phase_logic/results/cifar10/hybrid_q2_lbp2/crop_flip_augmentation/best-compact_shared_head2_v3_plain_w92400x92400_cuda_c16_explore800000_resample100_freeze200
