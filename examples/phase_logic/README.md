# Phase-Logic Image Classification

This example package trains differentiable logic networks on MNIST, CIFAR-10,
and CIFAR-100. It provides configurable input encoders, convolutional logic
backbones, dense logic heads, training treatments, and reproducible result
directories.

## Package layout

- `config.py`: experiment configuration and result-directory naming.
- `data.py`: dataset loading, encoding, and CIFAR augmentation.
- `model.py`: model builders for all supported datasets and model variants.
- `training.py`: training and evaluation routines.
- `losses.py`: cross-entropy and phase regularization losses.
- `schedules.py`: temperature and learning-rate schedules.
- `run_experiment.py`: command-line entry point for a single experiment.
- `compare.py`: aggregate and compare completed runs.
- `compare_rewiring.py`: compare matched fixed and loss-probed runs.
- `rewiring.py`: optional CIFAR-10 loss-probed connection updates.
- `compact.py`: CompactLogic candidate layers, CPU reference, and entropy controller.
- `classrouted_compact.py`: channel-concatenation skips with a fully shared 80k -> 80k CIFAR-10 head (historical builder name).
- `wire_erasing.py`: optional Boolean-wire erasure module.

## Datasets and input encodings

Supported datasets:

- MNIST: 10 classes.
- CIFAR-10: 10 classes.
- CIFAR-100: 100 classes.

Available input encodings:

- `fixed`: fixed-threshold Boolean channels.
- `quantile`: data-driven quantile thresholds.
- `sinusoidal`: random sinusoidal phase channels.
- `hybrid`: concatenated quantile and sinusoidal channels.
- `thermometer`: uniformly spaced thresholds, available for the CompactLogic variant.

The encoded channel count is derived from the selected encoding settings and
passed to the model through `encoder.output_channels`. For example:

- Q2 + S2 produces 12 channels for RGB images.
- Q4 produces 12 channels.
- Q4 + S4 produces 24 channels.

Optional Q2 + LBP2 encoding uses
`--input-encoding hybrid --hybrid-features lbp --num-quantile-bits 2`.
Omit `--num-sine-bits`: this replaces the sine branch, not the quantile branch.
For each RGB channel the output is `[Q1, Q2, right >= left, down >= up]`,
using radius-1 comparisons and replicated border padding. These are two
center-symmetric LBP-inspired bitplanes, not a full eight-neighbor LBP code
or histogram. The encoder has no trainable parameters and outputs 12 channels.
Quantile thresholds still come only from training data; local comparisons are
computed on each transformed image, in both training and evaluation.
Results use a separate `hybrid_q2_lbp2/` directory. Existing hybrid commands
keep their sine encoding and original result paths.

## Models

`examples/phase_logic/model.py` intentionally contains multiple model builder
functions. It acts as the model registry used by `--model-variant`, allowing
experiments to select a model without changing the training code.

MNIST builders:

- `build_model`
- `build_model_linear_head`

CIFAR-10 builders:

- `build_model_cifar10`
- `build_model_cifar10_org`
- `build_model_cifar10_smallmodel`
- `build_model_cifar10_compact_small`
- `build_model_cifar10_small_modifieddense`
- `build_model_cifar10_small_classrouted`
- `build_model_cifar10_small_classrouted_compact_head`
- `build_model_cifar10_small_compact_head`
- `build_model_cifar10_small_mrlb_compact_head`
- `build_model_cifar10_small_classrouted_channelmixed`
- `build_model_cifar10_deepconv`
- `build_model_cifar10_linear_head`

CIFAR-100 builders:

- `build_model_cifar100_small_classrouted`
- `build_model_cifar100_small_classrouted_channelmixed`

The `channelmixed` variants are opt-in ablations. They apply
`channel-balanced` connectivity only to the first LogicConv layer; later
LogicConv layers retain the original `random-unique` connectivity.

The current CIFAR-100 class-routed head is:

```text
4608 convolution outputs
  -> Dense1: 60,000 shared outputs
  -> Dense2: 60,000 shared outputs
  -> Dense3: 60,000 class-routed outputs
  -> Dense4: 30,000 class-routed outputs
  -> GroupSum(k=100)
```

All logic layers use rank-2 LUTs.

### Optional CompactLogic-style CIFAR-10 small model

`build_model_cifar10_compact_small` follows the upstream
[CompactLogic-S configuration](https://github.com/eth-sri/CompactLogic/blob/019677a79a1a3039e3c2c7f26689f3bcec8e22f9/experiments/configs/conv_cifar_S.yaml):

```text
RGB -> 3 uniform thresholds per color (9 binary channels)
  -> 3x3 conv: 128 channels, stride 2, output 128x16x16
  -> 3x3 conv: 128 channels, stride 1, output 128x16x16
  -> 3x3 conv: 512 channels, stride 2, output 512x8x8
  -> 3x3 conv: 512 channels, stride 1, output 512x8x8
  -> flatten 32,768 -> dense 80,000 -> dense 80,000
  -> GroupSum(k=10, tau=40)
```

Each neuron/kernel trains a softmax over 16 `(Boolean gate, input A, input B)`
candidates; discrete inference selects one two-input gate. Every spatial kernel
uses one input channel. The model has **2,580,480 training parameters** and
**291,072 unpruned Boolean node applications** at inference. Its layer widths
are `32768, 32768, 32768, 32768, 80000, 80000`. Parameter counts include all
candidate logits and are not the same measure as inference Boolean nodes.

Use `--parametrization compact --experiment pure_ce`. During the first 400,000
steps, every logic layer monitors per-row entropy each step (EMA 0.99,
tolerance 0.0005, patience 100). Stable rows with maximum candidate probability
above 0.95 preserve their winning triple and refresh other candidates. Rows
below 0.4 restart all candidates. Changed rows have Adam moments reset. This
mode does not use `--rewire-loss-probe` or its row/candidate budgets.

Optional `--compact-layer-aware-resampling` keeps 75% uniform opcode sampling
and draws 25% from AND/OR in Conv1–3 or XOR/XNOR in Dense2. Conv4 and Dense1
remain uniform. This applies only to replacement candidates during exploration;
initialization, strong winners, input-pair routing, Adam resets, and freezing
are unchanged. All 16 operations remain possible. Supported variants are
`build_model_cifar10_compact_small`,
`build_model_cifar10_small_classrouted_compact_head`, and
`build_model_cifar10_small_compact_head`, with either backend and either conv
freezing mode. No added parameters or inference gates. Runs use the separate
`layer_aware_resampling_v1_preferred25` results subdirectory; omit the flag to
retain the original behavior and paths. This is an experimental prior, not a
guarantee of better accuracy.

Separate opt-in **Efficient Layer-Aware Resampling (ELAR)** uses
`--compact-efficient-layer-aware-resampling` instead of the original flag:

- Conv1–3: 75% uniform + 25% equally split AND/OR (1, 7).
- Conv4: 100% uniform.
- Dense1: 70% uniform + 18% identity A/B (3, 5) + 8% NOT A/B (12, 10)
  + 4% TRUE/FALSE (15, 0).
- Dense2: 80% uniform + 12% identity A/B + 6% NOT A/B + 2% TRUE/FALSE.
  Each non-uniform category is split equally between its two operations.

These are sampling-mixture proportions, not guaranteed final gate frequencies;
uniform sampling can also select preferred operations or constants.
All 16 operations remain possible. ELAR has the
same supported variants and affects replacement opcodes only, preserving strong
winners, input routing, initialization, Adam resets, widths, and freezing.
It works with either backend and conv freezing mode, including optional dense
group freezing. The two layer-aware flags cannot be combined. With neither flag,
original uniform resampling is unchanged; the old flag's policy is unchanged.
ELAR runs use `efficient_layer_aware_resampling_v2_conv25_dense1_u70_i18_n8_c4_dense2_u80_i12_n6_c2`.
The efficiency benefit is experimental and must be established by synthesis.

Separate **aggressive layer-aware resampling** is enabled with
`--compact-aggressive-layer-aware-resampling`. It preserves ELAR-v2 and the
original layer-aware flag unchanged. Its replacement-opcode mixtures are:

| Layer | Uniform (all 16 ops) | Identity A/B | NOT A/B | TRUE/FALSE | AND/OR |
|---|---:|---:|---:|---:|---:|
| Conv1–3 | 75% | — | — | — | 25% |
| Conv4 | 100% | — | — | — | — |
| Dense1 | 50% | 30% | 5% | 15% | — |
| Dense2 | 60% | 25% | 5% | 10% | — |

Each preferred pair splits its category equally. These are sampling-mixture
weights, not final selected-gate frequencies; uniform sampling also includes
preferred operations. More constants may remove useful information and hurt
accuracy. Neither fewer gates nor improved FPGA routing is guaranteed.
Only one of the three layer-aware flags may be enabled. The same four-conv,
two-dense CIFAR-10 variants are supported, including wide75, with `none`,
`entropy`, or `hybrid` freezing and optional compatible dense-group freezing.
Only replacement opcodes change: initialization, input routing, strong winners,
entropy triggers, Adam resets, and model size are unchanged. Runs use
`aggressive_layer_aware_resampling_v1_conv25_d1_u50_i30_n5_c15_d2_u60_i25_n5_c10`.

After exploration stops, convolutional layers discretize and freeze in order
when their mean entropy stabilizes (patience 200). The two dense layers continue
training. Defaults are exposed through `--compact-exploration-steps`,
`--compact-resample-patience`, and `--compact-freeze-patience`.
Training uses Adam, constant LR 0.02, gradient-norm clipping at 10, and batch 128.
For this variant, `--cifar-augmentation` matches upstream training augmentation:
horizontal flip, edge-padded crop with padding 2, then `ToTensor()`.
Uniform encoding uses strict `>` thresholds and threshold-major RGB ordering.
Existing quantile/hybrid encoders remain available as separate experiments.

CUDA training uses the optional upstream MIT-licensed implementation, pinned
to commit `019677a79a1a3039e3c2c7f26689f3bcec8e22f9`. Install it into the active
environment with CUDA-enabled PyTorch and a matching CUDA compiler:

```bash
python -m pip install --no-build-isolation '.[compact]'
```

`--compact-backend auto` selects this backend on CUDA and the local PyTorch
reference on CPU. CUDA auto-selection reports a missing-extension error rather
than silently switching to a slow implementation. `--compact-backend torch`
explicitly selects the reference implementation; use it for small correctness
checks, not the 600K-step benchmark. The new layers support training/evaluation
and state dictionaries; TorchLogix circuit export is not integrated yet.

Results include `compact.csv` with cumulative resampling and frozen-layer counts.
With `--save-model`, `best_model.pt` contains the best observed discrete-test
checkpoint (model, optimizer, controller, config, metrics, and Torch RNG states).
`final_model.pt` retains the usual final model-state format. Frozen flags survive
state-dictionary loading and subsequent `train()` calls. Test evaluation uses all
10,000 images; upstream's loader drops the last incomplete test batch. These
implementation checks do not establish a reproduction of the published accuracy.

### CompactLogic channel-concatenation skips with a shared candidate head

`build_model_cifar10_small_classrouted_compact_head` combines the existing hybrid
encoder and two fully shared 80,000-wide dense candidate layers with a skip-connected
CompactLogic-S backbone: four single-operation 3x3 convolutions, padding 1, output channels
`128, 128, 512, 512`, and strides `2, 1, 2, 1`. Each kernel sees one input
channel and learns 16 gate/input candidates. It uses the upstream CUDA
`ConvLayer` implementation with `--compact-backend cuda`.

```text
hybrid encoder -> Conv1: 128x16x16 -> Conv2: 128x16x16
  -> concat(Conv1, Conv2): 256x16x16 -> Conv3: 512x8x8
  -> Conv4: 512x8x8
  -> concat(Conv3, Conv4): 1024x8x8 -> flatten: 65,536 shared features
  -> Dense1: 80,000 outputs, full 65,536-feature input pool
  -> Dense2: 80,000 outputs, full 80,000-feature input pool
  -> GroupSum(k=10, tau=40)
```

The historical builder name is retained, but there are no class-local restrictions.
Each dense stage uses one upstream candidate engine. The two skips concatenate
same-resolution feature maps without addition, projection, pooling, or detaching
gradients. Conv3 retains the upstream single-channel kernel restriction: kernels
cover both channel banks, but an individual kernel does not combine the banks.
Dense1 can combine shallow/deep features across its full input pool.

Concatenation adds no Boolean gates itself, but increases activation storage,
connection-index storage and hardware routing requirements; it is not zero total
hardware cost. It preserves a shorter feature/gradient path, not a guarantee
against vanishing gradients or ignored skip features. These remain sparse Boolean
candidate layers, not floating-point linear layers. Every conv and dense engine
uses the existing entropy monitoring, winner-preserving resampling, selective Adam
resets, and progressive convolution freezing.

Training parameters: **2,580,480** (20,480 convolution candidate logits plus
2,560,000 dense candidate logits). Output-wire widths are
`32768, 32768, 32768, 32768, 80000, 80000`, with **291,072 unpruned
inference Boolean node applications**, excluding encoding and GroupSum. These
counts and `GroupSum(k=10, tau=40)` match `build_model_cifar10_compact_small`.
Candidate logits are a training cost;
each conv kernel and dense output selects one candidate at discrete inference.

This variant uses the original CIFAR crop/flip augmentation (reflection padding
4). Choose `--parametrization compact --experiment pure_ce`; all candidate
mixtures use unit-temperature softmax. Both conv and dense resampling stop after
`--compact-exploration-steps`, then the four conv layers progressively discretize
and freeze while the dense head keeps learning. The previous loss-probed rewiring
flag is separate and unsupported here. Its results use a versioned
`compact_shared_head_v5_concat_w80000x80000_.../` directory, keeping earlier results
separate. Start a fresh run: previous checkpoints have incompatible shapes and/or
topology, even when parameter counts match.
Results include `compact.csv` and optional best/final checkpoints. The original
CIFAR builders and `build_model_cifar10_compact_small` remain unchanged.
Accuracy improvements are unmeasured.

### Small shared head with two 5%-wider dense stages

`build_model_cifar10_small_compact_head` keeps the original four-conv
`128,128,512,512` backbone and both concatenation skips, without MRLB or
class-local routing. The shared head is
`65536 -> 92400 -> 92400 -> GroupSum(k=10, tau=55)`.
Both dense stages are 5% wider than the 88,000-output baseline. The extra
44,000-output third layer has been removed. The same 16-candidate mechanism and
full shared input pools are retained. Final class groups contain 9,240 outputs;
`tau` remains unchanged.

Training parameters: **2,977,280**. Boolean node evaluations/image: **315,872**,
excluding encoding and GroupSum. Existing four-conv freezing schedules apply;
both dense stages remain trainable unless dense group freezing is enabled.
Results use the separate `compact_shared_head2_v2_concat_w92400x92400_.../` path;
old three-dense results are preserved.
Existing builders, including MRLB, remain available and unchanged. Select it
with `--model-variant build_model_cifar10_small_compact_head`; other flags are
unchanged. No accuracy improvement is guaranteed.

### Separate model with 75%-wider convolutions and dense stages

`build_model_cifar10_small_compact_head_wide75` widens all four convolution
stages to `224,224,896,896` and each dense stage to 161,700 outputs (+75% each).
The shared head is `114688 -> 161700 -> 161700 -> GroupSum(k=10, tau=55)`.
It preserves the encoder, strides, spatial kernels, both skips,
16 candidates per gate, and full shared routing. Training parameters:
**5,210,240** (+75% overall).
Boolean node evaluations/image: **552,776**, excluding encoder and GroupSum.
Existing entropy/hybrid conv freezing, optional dense group freezing, and either
layer-aware resampling flag are supported. Results use the separate
`compact_shared_head2_wide75_v2_concat_c224x224x896x896_w161700x161700_...` path.
Other models are unchanged; wider models need fresh training and more training
memory. Older dense-only wide75 results remain in their separate `v1` path.

### Optional layer-ordered dense group freezing

Add `--compact-dense-freeze` to a CompactLogic run. This is off by default;
existing entropy/hybrid convolution freezing, topology, and result paths stay
unchanged when it is omitted. It applies to shared CompactLogic dense heads.

After **all convolutions** (including MRLB branches, when present) have frozen,
process Dense1, then Dense2, then Dense3 when present. Select the lowest-entropy
unfrozen rows in groups of `--dense-freeze-fraction` of the original layer width
(default **0.05**). Entropy only ranks proposals; it does not establish safety.
For the current 92.4k/92.4k head, full groups contain 4,620 gates per layer.

Each proposal pins the selected gate **and input pair**, together with its logit
snapshot, across confirmation probes. Gates outside the accepted groups remain
trainable. Defaults:

| Flag | Default | Meaning |
| --- | ---: | --- |
| `--dense-freeze-min-interval` | 1000 | Recovery after conv completion and each committed group |
| `--dense-freeze-probe-interval` | 1000 | Probe scheduling interval |
| `--dense-freeze-probe-batches` | 4 | Independent unaugmented training mini-batches per probe (minimum 2) |
| `--dense-freeze-probe-patience` | 3 | Consecutive passing probes of the same pinned proposal |
| `--dense-freeze-max-loss-increase` | 0.01 | Maximum CE increase per batch and pooled upper bound |
| `--dense-freeze-loss-sigma` | 2.0 | Standard-error multiplier for the pooled bound |

Each batch's mean CE increase must pass, as must
`mean(CE_proposed-CE_current) + sigma*std(delta)/sqrt(N)` across all examples.
This bound is a heuristic, not a generalization guarantee. Probes do **not** use
Taylor scores, test examples, or validation accuracy. They replay Torch RNG and
restore model buffers between paired passes, and restore buffers, module modes,
and Torch RNG afterward. Rejections discard the proposal without changing the
network. No optimizer step occurs inside a probe.

Accepted rows execute their saved hard Boolean gates during training, including
after `model.train()`. Optimizer pre/post hooks protect their logits from Adam
momentum and weight decay and zero their `exp_avg`, `exp_avg_sq`, and AMSGrad
moments. They cannot be resampled. The remaining rows/layers continue adapting;
the next dense layer is not eligible until the current one is fully frozen.
Parameters and inference gate counts are unchanged, but masks, logit snapshots,
paired probes, and mixed hard/soft execution add training memory/work. This is
not currently a sparse-compute speed optimization.

At defaults, 40 groups for two layers take approximately **120k steps after
conv completion even if every probe passes**. Failed probes can delay this.
There is no forced completion; allow more iterations if needed. If conv freezing is
disabled (for example using a very large conv recovery interval), dense freezing
also never starts. Once everything is frozen, the loop skips optimizer updates
but retains scheduled evaluations.

Runs have a separate `dense_groups_v1_.../` path and `dense_freeze_probes.csv`.
Best checkpoints include `dense_freezer_state` alongside the model and Adam
state. Frozen masks and gate/pair snapshots are in the model state dictionary;
call `enable_dense_group_freezing(model)` **before** loading such a checkpoint.
Do not load these states into an unwrapped baseline model. The CLI does not
currently implement checkpoint resume. Accuracy benefits remain unmeasured.

### MRLB-v2 relational bridge

`build_model_cifar10_small_mrlb_compact_head` is a separate variant of the
current four-conv skip model. It keeps Conv1–Conv4 at `128,128,512,512`, both
shared dense stages at `88000`, the original concatenation paths, and the
baseline's current `GroupSum(k=10, tau=55)`.

```text
Conv1 (128x16x16) -> Conv2 (128x16x16)
  Conv2 -> Local: 64x16x16, rank-2 gates, radius 1
  Conv1 + Conv2 -> CrossDepth: 64x16x16, rank-2 gates, radius 2
concat(Conv1, Conv2, Local, CrossDepth): 384x16x16
  -> Conv3: 512x8x8 -> Conv4: 512x8x8
concat(Conv3, Conv4) -> flatten 65536 -> Dense 88000 -> Dense 88000 -> GroupSum
```

Local candidates take both operands from Conv2; cross-depth candidates always
take operand A from Conv1 and operand B from Conv2. These are learned Boolean
functions, not fixed subtraction/XOR/comparator operations. Both branches use
16 candidate triples per output channel, ordinary unit-temperature softmax,
and the existing upstream CUDA convolution kernels (3x3 and 5x5 respectively).

`--mrlb-conv3-relational-channels 128` reserves 128 of Conv3's 512 output gates:
64 take A from Local and 64 take A from CrossDepth, with B from the original
Conv1+Conv2 pool. The other 384 gates use only original features. Reserved rows
are spread across the outputs. The budget accepts even integers from 2 to 510
and is split equally between the branches. A winning identity/constant gate can
still ignore an operand: routing guarantees candidate access, not feature use.
These channel boundaries survive weak/strong resampling; strong winners are
preserved and changed Adam rows reset as before. No upstream package edits are
needed. The PyTorch reference also supports radius 2.

Both freezing controllers discover six spatial layers, in dependency order:
`Conv1, Conv2, Local, CrossDepth, Conv3, Conv4`. Dense layers remain trainable.
Reports include the branch/routing budget; results use a separate
`compact_mrlb_v2_local64r1_cross64r2_route128_w88000x88000_.../` directory.

Training parameters: **2,838,528** (baseline +2,048). Boolean node evaluations
per image: **339,840** (baseline +32,768, about 10.7%), excluding the encoder
and GroupSum. Extra activation storage, candidate buffers, and FPGA routing
are not captured by these counts. No accuracy improvement is established yet.

### Matched-budget six-convolution model

`build_model_cifar10_compact_sixconv` is a separate model; the existing four-conv
builders remain unchanged. It uses output channels `128,128,256,256,256,256`,
strides `2,1,2,1,1,1`, rank-2 gates, 16 candidates, and the selected input encoder.
It is a plain sequential stack, with **no skips or concatenation**:
`encoder -> six convs -> flatten -> Dense1 -> Dense2 -> GroupSum`.
Both dense stages are fully shared and 80,000-wide, ending in `GroupSum(k=10, tau=40)`.

Training parameters remain **2,580,480** and Boolean gate evaluations/image remain
**291,072** before pruning, excluding encoder and GroupSum. Logic output widths
are `32768,32768,16384,16384,16384,16384,80000,80000`. The final convolutional
feature pool is **16,384**, compared with 65,536 in the unchanged four-conv skip model.
Matching parameter/BOP counts does not imply identical representational capacity.
Results use a separate `compact_shared_head6_v2_plain_w80000x80000_.../` directory.
Start fresh; the previous concatenated six-conv checkpoints have incompatible
input shapes, even though their parameter counts match.

### Optional hybrid freezing (all CompactLogic variants)

`--compact-freeze-mode none` disables convolution freezing and hybrid probes;
all layers remain trainable after exploration. Connection resampling still
ends at `--compact-exploration-steps`. Discrete evaluation/export still selects
hard gates, without freezing training parameters. This mode cannot be combined
with `--compact-dense-freeze`. Runs use a separate `no_freeze_v1` directory.
There is no need to set an artificially long freezing interval.

`--compact-freeze-mode entropy` is the default and preserves the original
controller and result paths. `--compact-freeze-mode hybrid` is a separate,
experimental controller with the same exploration/resampling phase, followed by
shallow-to-deep whole-convolution freezing. Dense stages remain trainable.

Hybrid v3 is **probe-driven**: noisy entropy or a few changing kernels no longer
prevent the loss probes from running. The following diagnostics are recorded but
are **not eligibility requirements**:

- Layer-mean entropy/EMA difference, plus an EMA of its absolute magnitude.
- Instantaneous winner-change fraction and its rolling mean over
  `--compact-freeze-patience` steps (default 200). The old diagnostic stable-step
  counter uses entropy difference below `0.0005` and winner changes at most
  `--compact-freeze-max-winner-change` (default `0.001`). It neither blocks probes
  nor resets confirmations. In original entropy mode, patience still controls
  the unchanged mandatory stability streak.

Winner signatures include the actual Boolean opcode and both input coordinates,
not just the candidate-slot index. Monitoring starts after resampling has ended.
Hybrid freezing requires these conditions:

- At least `--compact-freeze-min-interval` steps (default **5,000**) have elapsed
  since the previous freeze, or since exploration ended for the first freeze.
- Every `--compact-freeze-probe-interval` steps (default **1,000**), an eligible
  whole layer is probed on `--compact-freeze-probe-batches` unaugmented **training**
  batches (default **4**), all at the same post-optimizer model snapshot. Graphs
  are released batch-by-batch; there are no optimizer updates between batches.
  **Every batch** must pass the first-order Taylor hardening-sensitivity limit
  `--compact-freeze-max-taylor` (default **0.05**) and the actual CE-increase limit
  `--compact-freeze-max-loss-increase` (default **0.01**). A favorable average
  cannot hide a failing batch.
- Compute paired per-example changes `d = CE_hard - CE_soft` over all probe
  batches. Require `mean(d) + sigma * std(d)/sqrt(N) <= max_loss_increase`, where
  `--compact-freeze-loss-sigma` defaults to **2.0**. This is an uncertainty-aware
  heuristic, **not a formal generalization guarantee**; zero disables the margin.
  Nonfinite results are rejected. A single example cannot estimate uncertainty
  and cannot pass with a positive sigma.
- `--compact-freeze-probe-patience` consecutive probes must pass (default **3**).
  Failed probes reset confirmations. Candidate changes reset them only if the
  fraction of changed kernels exceeds `--compact-freeze-max-probe-drift` (default
  **0.02**, or 2%). Drift is compared with the **first successful candidate in the
  streak**, not merely the previous probe, so small changes cannot accumulate
  beyond the limit unnoticed. Each probe still checks the entire current layer's
  actual hardening effect. These checks are evidence across nearby assignments,
  not repeated certification of one identical assignment. Probes use no test data,
  do not accumulate parameter gradients or update optimizer state, and restore
  model modes/RNG state. They add periodic training overhead.

The Taylor score is `mean_examples(abs(dot(grad_example, hard_output-soft_output)))`,
where each inner product sums the layer's channels and spatial positions.
It estimates a whole-layer perturbation, not pruning importance, and does not
guarantee safety; the direct loss probe is an additional check. Probe thresholds
are experimental, not accuracy-validated defaults. There is **no forced-freeze
fallback** if a layer never meets them.

Each hybrid run adds a settings-specific `hybrid_freeze_v3_probedriven_.../` path component,
leaving entropy-mode results separate. `freeze_probes.csv` records probe scores,
loss changes, worst-batch scores, pooled uncertainty margins, confirmations, and
decisions. Regular evaluations report the next layer and explicitly label the
stable-step count as diagnostic, alongside rolling winner changes and smoothed
entropy drift. Best checkpoints include hybrid controller state for restoration. Old
entropy-mode checkpoints/controllers are not interchangeable with hybrid state.
This remains whole-layer freezing, without partial-freeze masks or automatic
rollback. Several training batches do not replace held-out validation, and no
accuracy improvement is guaranteed. Prior v1/v2 hybrid results remain separate.

For a schedule smoke test, `--compact-exploration-steps 20000` makes the first
probe eligible at step **25,000** with default spacing/interval. The first freeze
can occur at **27,000** if all three checks pass. Allow more than 27,000 total
iterations (for example 80,000 to exercise the six-layer schedule); no freeze is
guaranteed if the loss/Taylor checks fail. This early exploration cutoff is for
testing the mechanism, not an established final-accuracy setting.

### Optional CIFAR-10 loss-probed rewiring

`--rewire-loss-probe` enables experimental rewiring. For
`build_model_cifar10_small_classrouted` it targets the first class-routed dense
layer; for `build_model_cifar10_smallmodel` it targets all four convolutional
and all three dense logic layers. The `--rewire-max-rows` budget is shared
across the small model's seven layers. After `--rewire-warmup` steps, every `--rewire-interval` steps
it checks row-level LightLUT entropy. Rows with low entropy or little entropy
change are eligible, but gates whose output already helps reduce loss receive
lower priority. For at most `--rewire-max-rows` eligible gates, it screens
all dense inputs using the minibatch output gradient, keeps the top
`--rewire-feature-pool` features, and forms `--rewire-candidates` diverse pairs
for the first dense layer. The later shrinking dense layers swap complete
input pairs between rows, preserving full input coverage. Conv kernels sample
candidate leaf pairs inside their existing receptive field while preserving
the kernel's original channel counts and distinct spatial inputs. The rewirer probes the best predicted pair
against downstream cross-entropy on both
the search batch and a separate, unaugmented training batch. A gate changes
only if both losses improve by more than `--rewire-min-improvement`. Small-model
rewires also must not worsen discrete cross-entropy on that layer's probe batch;
each layer uses a separate probe batch. Replaced LUT weights and corresponding
Adam moments are reset; dense pair swaps retain LUT weights and reset the two
rows' Adam moments. No extra model parameters are
introduced. The mode currently requires either CIFAR-10 model above,
`light` parametrization, and `pure_ce` or `annealing`; it does not use
Allen-Cahn phase loss.

Rewired runs use `rewire_loss_probe_v2_.../` for the class-routed model and
`rewire_loss_probe_v4_all_layers_safe_.../` for the small model, keeping their results
separate. Each writes
`rewiring.csv` with eligible, attempted, and accepted gate counts and probe
time; `summary.json` also records total training and rewiring time. Compare
fixed and rewired runs with identical seeds and training settings using
`compare_rewiring.py`. Accuracy gains have not been established yet.

## Training treatments

The `--experiment` option selects the optimization treatment:

| Treatment | Description |
| --- | --- |
| `pure_ce` | Cross-entropy only, with temperature fixed at 1.0. |
| `annealing` | Cross-entropy with linear temperature annealing. |
| `annealing_phase` | Annealing plus a phase regularizer. |

The phase-regularized loss uses the raw LightLUT weights:

```text
loss = cross_entropy
     + phase_weight * mean(0.25 * ((weight / margin)^2 - 1)^2)
```

The phase term has stable points at `-margin` and `+margin`.

Learning-rate schedules:

- `constant`: keep the initial learning rate for the full run.
- `cosine`: decay from `--learning-rate` to `--end-learning-rate`.

## CIFAR augmentation

`--cifar-augmentation` enables the existing training-only augmentation:

1. `RandomCrop(32, padding=4)`
2. `RandomHorizontalFlip()`
3. `ToTensor()`

`--cifar-random-erasing` is a separate opt-in addition. It keeps the crop and
flip operations and applies the following operation after `ToTensor()`, only
to the training data:

```python
RandomErasing(
    p=0.5,
    scale=(0.02, 0.12),
    ratio=(0.5, 2.0),
    value=0.0,
)
```

## Where results are saved

By default, runs are saved under:

```text
examples/phase_logic/results/
```

The exact directory is derived from the configuration. A typical CIFAR run is
stored as:

```text
examples/phase_logic/results/
  <dataset>/
    <encoding>/
      [crop_flip_augmentation/]
      [random_erasing/]
      [cosine_lr_.../]
      [rewire_loss_probe_.../]
      <parametrization>/
        <model_variant>/
          <experiment>/
            seed_<seed>/
```

Each run can contain:

- `config.json`: complete experiment configuration.
- `model_report.json`: parameter counts and logic-layer widths.
- `metrics.csv`: periodic training and evaluation metrics.
- `summary.json`: final and best reported metrics.
- `final_model.pt`: final checkpoint when `--save-model` is enabled.

Comparison summaries are written beneath the output root in a
`comparisons/` directory.

Use `--output-dir PATH` to save results somewhere else.

## Current progress

- MNIST training supports fixed, quantile, sinusoidal, and hybrid encodings.
- CIFAR-10 and CIFAR-100 use dynamic encoder channel counts.
- CIFAR class-routed models keep the convolutional backbone shared and route
  later dense stages by class.
- The strongest recorded CIFAR-10 result in the project notes is 65.87%
  discrete accuracy with Q2 + S2 hybrid encoding and seed 42421239.
- The channel-mixed variants remain optional because they have not yet shown a
  consistent accuracy improvement.

## Plot selected Boolean operations from a checkpoint

The separate, CPU-only plotting CLI reads either `best_model.pt` (checkpoint
dictionary) or `final_model.pt` (raw state dictionary), without constructing the
model or importing CompactLogic's CUDA extensions. It requires Matplotlib:
`python -m pip install '.[plots]'` (also listed in experiments/requirements.txt).

```bash
python -m examples.phase_logic.plot_gate_histogram \
  --checkpoint relative/path/to/best_model.pt \
  --output examples/phase_logic/plots/gate_histogram_3d.png \
  --frequency percent
```

Paths are relative to the terminal's working directory; absolute paths work too.
Without `--output`, the image is saved next to the checkpoint as
`best_model_gate_histogram_3d.png` (or the corresponding final-model name).
A matching `.csv` always records all 16 operation counts and percentages per
layer. PNG, PDF, and SVG outputs are supported; `--dpi` defaults to 180.

The 3D axes show **Boolean operation**, **layer**, and **frequency**. Use
`--frequency count` (default) for raw counts or `percent` to compare conv and
dense layers of very different sizes. Operation selection first finds the winning
candidate slot, then looks up its actual `gate_sequence` opcode; resampling may
assign the same Boolean operation to multiple slots. Pinned dense-row choices
are honored. Conv/branch frequencies count output kernels; dense frequencies
count output gates. Spatial reuse is excluded: these are **not BOP/image** counts.
Only CompactLogic candidate checkpoints are currently supported. Checkpoints
are read using `weights_only=True`; the CLI never falls back to unrestricted
pickle loading. An incomplete checkpoint being written by training may need a
retry. The original checkpoint is never modified.

## Training commands

Run commands from the repository root.

Show all available options:

```bash
python -m examples.phase_logic.run_experiment --help
```

### MNIST

Pure cross-entropy:

```bash
python -m examples.phase_logic.run_experiment \
  --dataset mnist \
  --experiment pure_ce \
  --parametrization light \
  --seed 22 \
  --device cuda
```

Temperature annealing:

```bash
python -m examples.phase_logic.run_experiment \
  --dataset mnist \
  --experiment annealing \
  --parametrization light \
  --seed 22 \
  --device cuda \
  --end-temperature 0.3
```

Annealing with phase regularization:

```bash
python -m examples.phase_logic.run_experiment \
  --dataset mnist \
  --experiment annealing_phase \
  --parametrization light \
  --seed 22 \
  --device cuda \
  --end-temperature 0.3 \
  --phase-lambda-max 0.01
```

Compare completed MNIST runs:

```bash
python -m examples.phase_logic.compare \
  --dataset mnist \
  --parametrization light \
  --seeds 22 42 123 2025 9999
```

### CIFAR-10

Class-routed model with Q2 + S2 hybrid encoding:

```bash
python -m examples.phase_logic.run_experiment \
  --dataset cifar10 \
  --model-variant build_model_cifar10_small_classrouted \
  --input-encoding hybrid \
  --num-quantile-bits 2 \
  --num-sine-bits 2 \
  --sine-sigma 1.0 \
  --sine-seed 0 \
  --cifar-augmentation \
  --experiment annealing \
  --parametrization light \
  --seed 42421239 \
  --device cuda \
  --iterations 200000 \
  --learning-rate 0.02 \
  --lr-schedule constant \
  --end-temperature 0.2 \
  --batch-size 128
```

Class-routed model with Q4 + S4 hybrid encoding:

```bash
python -m examples.phase_logic.run_experiment \
  --dataset cifar10 \
  --model-variant build_model_cifar10_small_classrouted \
  --input-encoding hybrid \
  --num-quantile-bits 4 \
  --num-sine-bits 4 \
  --sine-sigma 1.0 \
  --sine-seed 0 \
  --cifar-augmentation \
  --experiment annealing \
  --parametrization light \
  --seed 4242 \
  --device cuda \
  --iterations 187000 \
  --end-temperature 0.3 \
  --batch-size 128
```

First-convolution channel-mixed ablation:

```bash
python -m examples.phase_logic.run_experiment \
  --dataset cifar10 \
  --model-variant build_model_cifar10_small_classrouted_channelmixed \
  --input-encoding hybrid \
  --num-quantile-bits 2 \
  --num-sine-bits 2 \
  --sine-sigma 1.0 \
  --sine-seed 0 \
  --cifar-augmentation \
  --experiment annealing \
  --parametrization light \
  --seed 42421239 \
  --device cuda \
  --iterations 200000 \
  --learning-rate 0.02 \
  --lr-schedule constant \
  --end-temperature 0.2 \
  --batch-size 128
```

Cosine learning-rate experiment:

```bash
python -m examples.phase_logic.run_experiment \
  --dataset cifar10 \
  --model-variant build_model_cifar10_small_classrouted \
  --input-encoding hybrid \
  --num-quantile-bits 2 \
  --num-sine-bits 2 \
  --sine-sigma 1.0 \
  --sine-seed 0 \
  --cifar-augmentation \
  --experiment annealing \
  --parametrization light \
  --seed 42421239 \
  --device cuda \
  --iterations 200000 \
  --learning-rate 0.02 \
  --lr-schedule cosine \
  --end-learning-rate 0.002 \
  --end-temperature 0.2 \
  --batch-size 128
```

### Matched CIFAR-10 rewiring experiment

First run the CIFAR-10 class-routed Q2 + S2 baseline with `--seed 42`, then
repeat the same command with `--rewire-loss-probe`. For example:

```bash
python -m examples.phase_logic.run_experiment \
  --dataset cifar10 \
  --model-variant build_model_cifar10_small_classrouted \
  --input-encoding hybrid \
  --num-quantile-bits 2 \
  --num-sine-bits 2 \
  --sine-sigma 1.0 \
  --sine-seed 0 \
  --cifar-augmentation \
  --experiment annealing \
  --parametrization light \
  --seed 42 \
  --device cuda \
  --iterations 200000 \
  --learning-rate 0.02 \
  --lr-schedule constant \
  --end-temperature 0.2 \
  --batch-size 128 \
  --rewire-loss-probe
```

Repeat both runs for several seeds. Point the comparison tool at the two
directories that contain their `seed_<number>/` folders:

```bash
python -m examples.phase_logic.compare_rewiring \
  --baseline-root PATH_TO_FIXED_ANNEALING_DIRECTORY \
  --rewired-root PATH_TO_REWIRED_ANNEALING_DIRECTORY \
  --seeds 42 43 44
```

### CompactLogic-style CIFAR-10 small baseline

Install the optional CUDA backend as described above, then run:

```bash
python -m examples.phase_logic.run_experiment \
  --dataset cifar10 \
  --model-variant build_model_cifar10_compact_small \
  --parametrization compact \
  --experiment pure_ce \
  --input-encoding thermometer \
  --num-input-bits 3 \
  --cifar-augmentation \
  --compact-backend cuda \
  --compact-exploration-steps 400000 \
  --iterations 600000 \
  --learning-rate 0.02 \
  --lr-schedule constant \
  --batch-size 128 \
  --eval-frequency 2000 \
  --num-workers 4 \
  --seed 0 \
  --device cuda \
  --save-model
```

An encoding ablation can replace `--input-encoding thermometer --num-input-bits 3`
with `--input-encoding hybrid --num-input-bits 1 --num-quantile-bits 4 --num-sine-bits 4`.
Keep the remaining settings and seeds matched.

### CIFAR-10 CompactLogic concatenation skips with shared candidate learning

Q4 + S4 keeps the encoding of the earlier class-routed experiment. This variant
uses two same-resolution concatenation skips and fully shared 80,000-wide Dense1
and Dense2. Match encoding, seed, and training settings when comparing against
the no-skip `build_model_cifar10_compact_small` baseline.

```bash
python -m examples.phase_logic.run_experiment \
  --dataset cifar10 \
  --model-variant build_model_cifar10_small_classrouted_compact_head \
  --input-encoding hybrid \
  --num-quantile-bits 4 --num-sine-bits 4 \
  --sine-sigma 1.0 --sine-seed 0 \
  --cifar-augmentation \
  --experiment pure_ce --parametrization compact \
  --compact-backend cuda --compact-exploration-steps 400000 \
  --iterations 600000 --eval-frequency 2000 \
  --learning-rate 0.02 --lr-schedule constant \
  --batch-size 128 \
  --num-workers 4 --seed 4242449 --device cuda --save-model
```

### CIFAR-10 six-conv comparison: original versus hybrid freezing

Run both with identical seed, encoder and training budget. The first retains the
original freezing method (not "no freezing"); the second opts into hybrid freezing.
Both use the plain (no-skip) six-conv stack, Q4 + S1, and the 80k -> 80k shared
dense head. Hybrid uses the four-batch/two-standard-error guard by default.

Original entropy freezing:

```bash
python -m examples.phase_logic.run_experiment \
  --dataset cifar10 --model-variant build_model_cifar10_compact_sixconv \
  --input-encoding hybrid --num-quantile-bits 4 --num-sine-bits 1 \
  --sine-sigma 1.0 --sine-seed 0 --cifar-augmentation \
  --experiment pure_ce --parametrization compact --compact-backend cuda \
  --compact-exploration-steps 800000 --iterations 990000 \
  --compact-freeze-mode entropy --eval-frequency 2000 \
  --learning-rate 0.02 --lr-schedule constant --batch-size 128 \
  --num-workers 4 --seed 4242449 --device cuda --save-model
```

Hybrid freezing (the only changed setting is the freezing mode):

```bash
python -m examples.phase_logic.run_experiment \
  --dataset cifar10 --model-variant build_model_cifar10_compact_sixconv \
  --input-encoding hybrid --num-quantile-bits 4 --num-sine-bits 1 \
  --sine-sigma 1.0 --sine-seed 0 --cifar-augmentation \
  --experiment pure_ce --parametrization compact --compact-backend cuda \
  --compact-exploration-steps 800000 --iterations 990000 \
  --compact-freeze-mode hybrid --eval-frequency 2000 \
  --learning-rate 0.02 --lr-schedule constant --batch-size 128 \
  --num-workers 4 --seed 4242449 --device cuda --save-model
```

To test freezing alone on the existing four-conv skip architecture, keep either
mode and replace only the model name with
`build_model_cifar10_small_classrouted_compact_head`. Use fresh runs and a validation
split for tuning before final test-set comparison; accuracy improvements are not
established by construction tests.

### CIFAR-100

Class-routed model with Q4 + S4 hybrid encoding and random erasing:

```bash
python -m examples.phase_logic.run_experiment \
  --dataset cifar100 \
  --model-variant build_model_cifar100_small_classrouted \
  --input-encoding hybrid \
  --num-quantile-bits 4 \
  --num-sine-bits 4 \
  --sine-sigma 1.0 \
  --sine-seed 0 \
  --cifar-augmentation \
  --cifar-random-erasing \
  --experiment annealing \
  --parametrization light \
  --seed 4242 \
  --device cuda \
  --iterations 200000 \
  --learning-rate 0.02 \
  --lr-schedule constant \
  --end-temperature 0.2 \
  --batch-size 128 \
  --save-model
```

First-convolution channel-mixed ablation:

```bash
python -m examples.phase_logic.run_experiment \
  --dataset cifar100 \
  --model-variant build_model_cifar100_small_classrouted_channelmixed \
  --input-encoding hybrid \
  --num-quantile-bits 4 \
  --num-sine-bits 4 \
  --sine-sigma 1.0 \
  --sine-seed 0 \
  --cifar-augmentation \
  --cifar-random-erasing \
  --experiment annealing \
  --parametrization light \
  --seed 4242 \
  --device cuda \
  --iterations 200000 \
  --learning-rate 0.02 \
  --lr-schedule constant \
  --end-temperature 0.2 \
  --batch-size 128 \
  --save-model
```
