# Progressive aggressive infusion for KD students

Enable with `--kd-progressive-aggressive-infusion` in `train_kd`. This isolated
implementation imports the five immutable policy mixtures from the original
`examples/phase_logic/infusion_training`; it does not change that implementation
or any `src/torchlogix` code.

The LUT6 student has one truth table per node, not CompactLogic's bank of
candidate two-input gates. Here infusion proposes replacement truth tables for
a bounded fraction of the weakest dense nodes, ranked by mean absolute table
value. Existing constant, identity, and inverted-identity functions are kept.
Convolution tables and input connection indices are not resampled. All
parameters that were trainable continue normal KD updates after a proposal.

| Level | Uniform random LUT6 | Identity of one pin | NOT of one pin | TRUE/FALSE |
| --- | ---: | ---: | ---: | ---: |
| 1 | 50% | 30% | 5% | 15% |
| 2 | 40% | 25% | 10% | 25% |
| 3 | 30% | 20% | 10% | 40% |
| 4 | 25% | 15% | 10% | 50% |
| 5 | 15% | 10% | 5% | 70% |

Pins are selected uniformly among the six inputs. TRUE and FALSE are equally
likely. Pin 0 remains the least significant LUT address bit. These percentages
describe newly proposed tables, not the whole network or the final trained
constant fraction.

## Initial experiment

Use the 61.9% run's KD loss settings with 1,500 epochs, as requested:

```bash
python -m examples.phase_logic.KD_training.train_kd \
  --mode kd --student-model tinyconv_q2_lbp2_dense2 \
  --device cuda --teacher-backend cuda \
  --epochs 1500 --batch-size 128 --learning-rate 0.02 \
  --seed 42 --alpha 0.5 --temperature 2 --student-tau 40 \
  --kd-progressive-aggressive-infusion
```

Infusion and `--discrete-stability-staged-freezing` are currently mutually
exclusive: their independent rollback checkpoints must not restore conflicting
controller states. The command above tests infusion on its own.

Defaults:

- `--kd-infusion-warmup-epochs 100`: complete 100 epochs before the first
  proposal; normal KD runs throughout warmup.
- `--kd-infusion-interval-epochs 10`: propose after validation at epochs
  100, 110, 120, etc. No proposals are made after the final epoch.
- `--kd-infusion-fraction 0.01`: replace at most 1% of each dense layer at
  each event (80 + 80 tables for the 8,000/8,000 model). Frozen tables are
  skipped. Adam row moments are cleared only for replaced tables.
- `--kd-infusion-cooldown-steps 15000`: minimum optimizer steps between
  level increases and before retrying proposals after rollback.

The controller starts at Level 1. Three consecutive validation evaluations
must stay stable in discrete accuracy (within 0.3 percentage points) and CE
loss (within 0.05), with accuracy near the best observed/safe reference, before increasing
one level. A safe checkpoint is written before the first proposal and before
every increase. An accuracy drop greater than 1.5 percentage points or CE-loss
increase greater than 0.10 from that safe checkpoint restores the student,
optimizer, controller level, and RNG state. Failed validation results cannot
overwrite `best_student.pt`. Proposals are applied after saving the evaluated
best model, so unevaluated tables cannot become the best checkpoint.

## Logs and resume

- `metrics.csv`: normal metrics plus the level used to produce that epoch's
  validation output; the terminal also prints the level.
- `kd_infusion.csv`: observations, advances, rollbacks, and per-layer proposal
  counts split by function family.
- `kd_infusion_safe.pt`: the latest safe point before an increase.
- `kd_infusion_latest.pt`: resumable training state after an epoch, including
  the safe point, best checkpoint, optimizer, RNGs, and data-loader generators.
- `summary.json`: controller state and actual trained constant-node count.

To resume, repeat the same command and append:

```bash
--kd-infusion-resume path/to/previous_run/kd_infusion_latest.pt
```

Use a total `--epochs` larger than the saved epoch. Resume creates a new run
directory, whose CSV files contain the resumed epochs; the earlier CSV files
stay in the previous run. Learning settings and infusion options must match.
Inference checkpoints (`best_student.pt` and `discrete_student.pt`) are not
resume checkpoints.

## Additional student architectures

Compatible dense table parameters are discovered by structure: names `tables`
or `*_tables`, shape `[nodes,64]`, in a module without convolutional
`in_channels`/`out_channels`. This supports root Dense1/Dense2 parameters and
nested dense modules, with no student-name allowlist in infusion. New students
still need the normal model registration in the KD trainer. Other LUT widths
or parameter layouts require a corresponding adapter.

This is an accuracy-versus-simplification experiment. Accuracy and FPGA savings
must be measured after training and export; candidate percentages guarantee
neither result.
