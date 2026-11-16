# Isolated CIFAR-10 LUT6 knowledge distillation

All implementation and output files stay in `examples/phase_logic/KD_training/`.
Existing ELAR models, checkpoints and training/export code are unchanged.
Nothing trains, downloads datasets, or installs packages on import.

For the optional five-level dense LUT6 infusion experiment, see
[KD progressive aggressive infusion](kd_infusion_training/README.md).

## Students

`--student-model lut6` remains the default and preserves the original student:

```
RGB [0,1], 3 x 32 x 32
  -> 10-bit uniform thermometer per pixel/channel (30,720 bits)
  -> one layer of 8,000 learned LUT6 nodes
  -> GroupSum: 800 Boolean outputs/class, 10 logits, divided by student_tau
```

Thresholds are `1/11, ..., 10/11`, comparison is strict `>`, flattening order
is channel/pixel/threshold. No normalization, CNN, extra dense layer, learned
classifier, or teacher features are deployed. Every LUT has 64 signed real
training entries; nonnegative entries become TRUE for Boolean inference.
Pin 0 is the least significant address bit. Training uses hard Boolean forward
passes too, with surrogate backward derivatives. The final model selects one
fixed input per pin and one Boolean value per table entry.

### Differences from original DWN

[DWN](https://github.com/alanbacellar/DWN) is not installed locally. Its current
learnable mapping uses a full input-by-pin parameter matrix and its LUT backward
uses a custom CUDA EFD operator. This self-contained PyTorch approximation uses:

- 16 fixed candidate inputs per pin, learned mapping logits, hard argmax forward
  selection and a softmax-mixture surrogate gradient (temperature configurable).
  Pools are sampled once and disjoint across the six pins of each node, ensuring
  six distinct selected inputs. Connections cannot search outside these pools.
- Hard-address truth-table lookup, multilinear derivatives at the binary input
  vertices, and an identity STE for output binarization. LUT weights are clamped
  to `[-1,1]` after updates. This is **not DWN's EFD training algorithm**.
- Uniform thermometer thresholds, rather than DWN's data-fitted distributive
  thermometer. No thresholds are fitted to validation/test data.

The bounded mapping is an experimental approximation, **not equivalent to DWN's
original learnable mapping**. This is not a reproduction of DWN's published
57.42% CIFAR-10 accuracy, and no student accuracy has been measured here.

Static autograd review: mapping gradients sum over the batch and apply the
candidate-axis softmax Jacobian. LUT table gradients accumulate repeated
addresses; pin gradients use `table[bit=1] - table[bit=0]`. The hard training
and evaluation paths use the same argmax mapping, address convention and
nonnegative-to-TRUE threshold. Encoded pixels deliberately receive no gradients.
These checks are source-level reasoning, not numerical gradient tests or
experimental evidence that this surrogate trains successfully.

At 16 candidates/pin there are 512,000 table parameters plus 768,000 mapping
parameters: **1,280,000 training parameters**. The deployed network has 8,000
LUT6 nodes, not 8,000 rank-2 BOPs. Initialization samples candidates across all
30,720 encoded inputs, without spatial/channel restrictions. Node chunks bound
temporary candidate-gather memory; this is not an optimized CUDA implementation.

### Optional tiny Boolean convolution student

Select `--student-model tinyconv_q2_lbp2` for:

```text
RGB 3x32x32
  -> existing HybridFeatureEncoder Q2+LBP2: 12x32x32
  -> Boolean LUT6 conv: 32 channels, 3x3, stride 2, zero padding 1: 32x16x16
  -> Boolean LUT6 conv: 64 channels, 3x3, stride 2, zero padding 1: 64x8x8
  -> flatten 4,096 bits
  -> existing shared 8,000-node LUT6 head
  -> GroupSum: 800 Boolean outputs/class, 10 logits
```

These are Boolean convolutions, not floating-point Conv2d layers. Each channel
shares one learned LUT6 and six selected patch inputs over all spatial positions.
The same bounded-candidate mapping and LUT/STE training surrogate apply to both
convolutions and the head. The input mapper propagates surrogate gradients into
the preceding convolutional features; original fixed encoded pixels need none.
There is no convolutional freezing, candidate resampling, or new CUDA kernel.
At 16 candidates/pin: **1,295,360 training parameters**, **8,096 shared LUT6
definitions**, and **20,288 LUT6 evaluations/image**, excluding the encoder and
GroupSum arithmetic. LUT6 evaluations are not rank-2 BOP counts.

The encoder is imported unchanged from the main work. Quantile thresholds are
fitted using the existing histogram helper on **only the student training split**
and saved as encoder buffers. LBP2 uses the same horizontal/vertical neighbor
comparisons, replicated border and channel ordering as the main encoder.
The teacher retains its own original thresholds and encoder, unchanged.
Conv1's disjoint six-pin candidate pools permit at most 18 candidates/pin;
the default remains 16. All final features and table outputs are binary.

Example KD run with the new student (automatic unique output directory):

```bash
python -m examples.phase_logic.KD_training.train_kd \
  --mode kd --student-model tinyconv_q2_lbp2 \
  --device cuda --teacher-backend cuda \
  --epochs 500 --batch-size 128 --learning-rate 0.01 \
  --seed 423 --alpha 0.5 --temperature 2.7
```

For a matched baseline use the same settings with `--mode baseline`.
`best_student.pt` contains the complete new student state, including encoder and
convolutions. Its `discrete_student.pt` additionally records the two convolution
truth tables, patch connections and geometry; no teacher parameters are saved.
Only source review and syntax checks were performed for this new variant;
successful training or improved accuracy has not been established.

### 10% wider one-dense variant

Use `--student-model tinyconv_q2_lbp2_wide10` for a separate widened version
of the one-dense student. Integer widths are rounded to Conv35/Conv70, and the
dense head is widened exactly from 8,000 to 8,800 LUT6 nodes:

```text
Q2+LBP2 -> LUT6 Conv35 -> LUT6 Conv70 -> 8,800 LUT6 -> GroupSum(10, 880/class)
```

At 16 candidates/pin it has 1,424,800 trainable parameters and 22,240 LUT6
evaluations/image, excluding the encoder and GroupSum. Existing students are
unchanged. Train it by replacing the model argument in the command above:

```bash
--student-model tinyconv_q2_lbp2_wide10
```

This new architecture requires training from scratch and is not supported by
the current two-dense post-training circuit/hls4ml exporters.

### Two-dense variant

Use `--student-model tinyconv_q2_lbp2_dense2` to keep the same Q2+LBP2 encoder
and two Boolean convolutions, followed by **8,000 LUT6 -> 8,000 LUT6 -> GroupSum**.
Both dense layers are shared, with the same bounded learnable input mapping.
Only the second dense layer feeds GroupSum (800 outputs/class); no intermediate
GroupSum is applied. Gradients propagate through both dense layers and convs.
At 16 candidates/pin this variant has **2,575,360 training parameters** and
**28,288 LUT6 evaluations/image**, excluding encoder and GroupSum arithmetic.
The two existing student variants and the default selection remain unchanged.
Best and discrete checkpoints include both layers. Replace the student-model
argument in the command above; all other CLI options remain available.
This variant has only been reviewed statically and syntax-checked, not trained.

### Wider two-dense Q2+LBP2 variant

`tinyconv_q2_lbp2_dense2_wide10` keeps the Q2+LBP2 encoder and Conv32/64
layers above, then uses two 8,800-node LUT6 dense layers and GroupSum with
880 outputs per class. It has 2,831,360 trainable parameters. To compare it
against the 61.9% two-dense run, use that run's training settings:

```bash
python -m examples.phase_logic.KD_training.train_kd \
  --mode kd --student-model tinyconv_q2_lbp2_dense2_wide10 \
  --device cuda --teacher-backend cuda --epochs 500 --batch-size 128 \
  --learning-rate 0.02 --seed 42 --alpha 0.5 --temperature 2 \
  --student-tau 40 --discrete-stability-staged-freezing 1
```

This starts a new training run. The two-dense post-training exporters currently
accept only their registered checkpoint architectures, so add this variant to
their loader before exporting it to circuit JSON or HLS.

### Q2-only compact two-dense variant

Use `--student-model tinyconv_q2_compact20_dense2` for the Q2-only version
with every trainable layer 20% narrower than `tinyconv_q2_wide10_dense2`:
Conv28/56, Dense7040/7040, and GroupSum(704 outputs/class). At 16
candidates/pin it has 2,265,064 trainable parameters and 24,832 LUT6
evaluations/image, both 20% below the Q2 wide two-dense variant. The
Q2+LBP2 `tinyconv_q2_lbp2_dense2` has 2,575,360 parameters and 28,288
evaluations/image, so this new model is about 12% smaller than that model.
The encoder remains Q2-only; existing checkpoints are unaffected, and the
new variant must be trained from scratch. The two-dense hls4ml JSON exporter
accepts its checkpoints through the registered loader.

### LBP2-only wide two-dense variant

Use `--student-model tinyconv_lbp2_wide10_dense2` for two fixed radius-1
comparisons per RGB channel (horizontal and vertical), without Q2 thresholds.
It has six Boolean input planes, Conv35/70, Dense8800/8800, and GroupSum with
880 outputs per class. The comparisons use replicate padding and treat ties as
TRUE, matching the LBP2 portion of the Q2+LBP2 encoder. This is a new model
that must be trained from scratch; it does not change existing checkpoints.
The two-dense hls4ml JSON exporter accepts its checkpoints.

### Optional discrete-stability staged freezing

For any two-dense student, provide the final exploration epoch. For example,
to keep all mappings trainable through epoch 500, add:

```bash
--discrete-stability-staged-freezing 500
```

This opt-in controller tracks the hard selected connection (`argmax`) for every
LUT6 pin beginning after the requested exploration epoch, and freezes connection
mappings in the order Conv1 then Conv2. Dense1 and Dense2 mappings remain
trainable. A convolution mapping must stay within an integer change budget for
10 consecutive epochs. The budget is
`max(1, floor(0.1% * mapping_count))`.
If a stage never reaches that condition, it makes a validation-protected freeze
attempt after 100 observation epochs rather than waiting indefinitely for exact
stability. A rejected attempt can still roll back and retry. Before freezing,
the trainer saves `staged_freezing_safe.pt`; it then monitors five
validation epochs and restores the safe model and optimizer if mean accuracy
drops by more than 1.5 percentage points or mean loss increases by more than
0.05. A failed stage waits 10 epochs before retrying. LUT truth tables remain
trainable throughout, so this stabilizes deployed connectivity without reusing
the incompatible CompactLogic soft/hard hybrid-freezing probe.

Per-epoch decisions, changed-mapping counts, allowed counts, and mapping sizes
are written to `staged_freezing.csv`. Runs using the flag
include `dsfreeze` in their automatically generated directory name. The flag is
rejected for the other student architectures.

## Teacher protection and scaling

`teacher.py` imports the existing `post_training.load_checkpoint` utility. The
default checkpoint is the ELAR-v2 + hybrid-freezing `best_model.pt`, seed 4242449,
inside the specified `results/bestof best/...explore700000...` directory.
Loading is read-only and strict. Saved encoder buffers, discrete gate choices,
original architecture and reported GroupSum tau are restored by that utility.
Configuration resolution matches the existing loader: prefer embedded checkpoint
`config` metadata, otherwise use sibling `config.json`. A missing JSON file is
acceptable when that metadata is present. The existing loader still requires a
saved model report with GroupSum tau for this wrapper to accept the scale without
guessing. Metadata/state structure is inspected on CPU before initialization;
architecture/weight compatibility is checked later by the strict model loader.
The teacher stays in discrete `eval()` mode, `requires_grad_(False)` and
`no_grad()`. No teacher state or optimizer is updated or saved.

Both networks receive the **same augmented RGB batch**, then apply their own
encoders. Teacher outputs already include its saved GroupSum scaling. Student
outputs from either variant use their own configurable `--student-tau` (default 40). Neither raw
counts nor previously softmaxed probabilities are substituted for these logits.
Different class-independent score offsets do not affect softmax; scale does.
The default student tau is an experimental setting, not a calibrated optimum.

KD uses `(1-alpha)*CE + alpha*T*T*KL(teacher || student)`, with `log_softmax`
for the student and `softmax` for the teacher. Defaults: alpha 0.5 and T 2.
There is no intermediate feature distillation.

## Run from the repository root

Use the existing environment with PyTorch, torchvision, NumPy and torchlogix.
CUDA teacher inference additionally needs the existing CompactLogic extension;
`--teacher-backend torch` uses its slower PyTorch reference instead. The student
itself needs no DWN package or custom kernels. CIFAR-10 must already exist in
`data/` (or pass `--data-dir`); all dataset construction uses `download=False`.

Baseline (labels only; inspects the checkpoint on CPU for configuration but does
not construct or run the teacher):

```bash
python -m examples.phase_logic.KD_training.train_kd \
  --mode baseline --device cuda --epochs 100 --batch-size 128 \
  --learning-rate 0.01 --seed 42
```

Logit KD, with the same student and preprocessing:

```bash
python -m examples.phase_logic.KD_training.train_kd \
  --mode kd --device cuda --teacher-backend cuda \
  --epochs 100 --batch-size 128 --learning-rate 0.01 --seed 42 \
  --alpha 0.5 --temperature 2.0
```

CPU execution: use `--device cpu --teacher-backend torch`. Inspect all options
with `--help`. Optional controls include `--student-tau`, `--mapping-temperature`,
`--mapping-candidates`, `--node-chunk`, `--no-augmentation`, `--student-model`,
`--discrete-stability-staged-freezing`, and `--teacher`.

## Validation and outputs

The original pipeline has no held-out validation split, so this experiment uses
a seeded 45,000/5,000 split of CIFAR-10 **training** images, configurable via
`--validation-size` and `--split-seed`. Both modes use identical initialization,
split, shuffle seeds, and RGB augmentation. Augmentation mirrors the teacher's
saved configuration (including whether augmentation was enabled and its
crop/flip ordering); `--no-augmentation` can explicitly disable it. Validation is ToTensor
only. CUDA reductions can still introduce numerical nondeterminism.

The teacher was originally trained on all 50,000 training images. This validation
holdout is therefore student-held-out, **not teacher-unseen**; teacher soft labels
can carry training-set knowledge. Interpret KD-vs-baseline validation accordingly.

Best checkpoints use **discrete validation accuracy**, breaking ties by validation
CE. Test data is never opened by default. `--evaluate-test` explicitly opts into
one final test evaluation of the best checkpoint after training; omit it during
development and hyperparameter selection.

Results automatically get a new directory under `KD_training/runs/`, named
with CIFAR-10, student model, mode, epochs, batch size, learning rate, candidate count, student
tau, seed, KD alpha/temperature (KD only), and the UTC date/time with microseconds.
For example:

```text
cifar10_lut6_kd_e500_b128_lr0.01_c16_tau40_seed42_a0.5_T2_20261028T120000_123456Z/
```

The resolved name is saved in `config.json`. Each launch generates a fresh name;
there is no need to supply `--run-name` unless you want a specific name.

- `config.json`, `split.pt`, `metrics.csv`, `summary.json`.
- `best_student.pt`: student state, optimizer, configuration and validation result.
- `discrete_student.pt`: only selected input indices, Boolean truth tables,
  thresholds, address convention and GroupSum scale/group size. This is data,
  not a self-contained executable model or FPGA export.

The two-dense student can be exported to native-LUT6 Verilog from either
`best_student.pt` or `discrete_student.pt`. See
[`post_training/README.md`](post_training/README.md) for the command and RTL
interface.

Existing run directories (including explicit `--run-name` overrides) are rejected
before loading the teacher, datasets or models. A requested best checkpoint is
validated first; the run directory is then reserved before dataset/model setup.
If later initialization fails, that empty or partial run directory may remain.
`--output-dir` must remain inside `KD_training/`. To continue any KD student
from its selected best epoch, repeat its original training settings, set
`--epochs` above the checkpoint epoch, and add
`--resume-from-best path/to/best_student.pt`. This creates a new run, restores
the student and Adam optimizer, and starts at the next epoch. Previously
accepted convolution freezes remain frozen. The best checkpoint does not
contain the original random/data-loader state or unfinished freeze/infusion
monitoring, so this is a continuation from best weights rather than an exact
replay of the interrupted run. For exact KD infusion epoch-boundary resume,
use `--kd-infusion-resume path/to/kd_infusion_latest.pt` instead.

There is no scheduler, automatic sweep, automatic in-training RTL export, or
promised accuracy/routing improvement. The native-LUT6 exporter is run separately after checkpointing.
No training, dataset download, GPU execution or FPGA benchmark is performed by
the exporter; it only loads and validates an existing student checkpoint.
