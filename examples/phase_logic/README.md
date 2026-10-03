# Phase-logic image-classification comparisons

This package runs controlled comparisons of LightLUT training on binary MNIST
and CIFAR-10 inputs. MNIST remains the default dataset.

## Datasets

- `mnist`: `1x28x28`, binarized as nonzero ink versus zero background
- `cifar10`: continuous `3x32x32` RGB tensors encoded inside the model as
  fixed, quantile thermometer, sinusoidal, or hybrid binary bitplanes

Select CIFAR-10 with:

```bash
python -m examples.phase_logic.compare --dataset cifar10 --seeds 42
```

Choose the CIFAR encoding explicitly:

```bash
# Three channels: one fixed threshold (0.5) per RGB channel
python -m examples.phase_logic.run_experiment \
  --dataset cifar10 --input-encoding fixed --num-input-bits 1

# Six channels: 33% and 67% training-pixel quantiles per RGB channel
python -m examples.phase_logic.run_experiment \
  --dataset cifar10 --input-encoding quantile --num-input-bits 2

# Twelve channels: 20%, 40%, 60%, and 80% quantiles per RGB channel
python -m examples.phase_logic.run_experiment \
  --dataset cifar10 --input-encoding quantile --num-input-bits 4

# Twelve channels: four fixed sinusoidal projections per RGB channel
python -m examples.phase_logic.run_experiment \
  --dataset cifar10 --input-encoding sinusoidal --num-sine-bits 4 \
  --sine-sigma 1.0 --sine-seed 0

# Twelve channels: two quantile plus two sinusoidal bits per RGB channel
python -m examples.phase_logic.run_experiment \
  --dataset cifar10 --input-encoding hybrid \
  --num-quantile-bits 2 --num-sine-bits 2 \
  --sine-sigma 1.0 --sine-seed 0
```

Quantile thresholds are computed once from the unaugmented training split,
stored as fixed model buffers, and reused unchanged for evaluation and test
data. Sinusoidal weights are sampled from `Normal(0, sigma^2)` with an isolated
encoder seed, phases are sampled uniformly from `[0, 2*pi)`, and both are fixed
model buffers. Hard sine binarization uses a straight-through estimator. The
run prints the thresholds/projections and encoder input/output shapes.

For the controlled four-bit-per-channel comparison, run `quantile` with four
quantile bits, `sinusoidal` with four sine bits, and `hybrid` with two of each.
All modes then present exactly 12 channels to the first logic convolution.

`--model-variant` accepts the exact builder function name from `model.py`:

- MNIST: `build_model`, `build_model_linear_head`
- CIFAR-10: `build_model_cifar10`, `build_model_cifar10_org`,
  `build_model_cifar10_smallmodel`, `build_model_cifar10_linear_head`

## Treatments

| Experiment | Training loss | Temperature |
|---|---|---|
| `pure_ce` | cross-entropy | constant at `1.0` |
| `annealing` | cross-entropy | `1.0` to `0.2` after warm-up |
| `annealing_phase` | cross-entropy + logit-margin phase loss | same annealing schedule |

The phase arm uses raw LightLUT logits directly:

```text
L_phase = mean(relu(margin - abs(weight))^2)
```

It linearly increases its coefficient from zero after the configured warm-up.
`pure_ce` uses unmodified `torch.nn.CrossEntropyLoss`. All three arms use the
parametrization selected with `--parametrization`, as well as the same model
architecture, seed, initialization, shuffled batch order, optimizer, and
evaluation procedure. The annealing arms differ only in whether logit-margin
phase regularization is enabled.

The `annealing_phase` method requires `--parametrization light`; the raw-logit
margin loss is intentionally specific to LightLUT.

## Run the paired comparison

Run commands from the repository root with the `logc_env` environment active:

```bash
python -m examples.phase_logic.compare --seeds 42 43 44
```

Choose one parametrization for every comparison arm:

```bash
python -m examples.phase_logic.compare --parametrization light --seeds 42 43 44
```

For a quick single-seed comparison:

```bash
python -m examples.phase_logic.compare --seeds 42
```

Choose one or more builders to compare:

```bash
python -m examples.phase_logic.compare \
  --dataset cifar10 \
  --model-variants build_model_cifar10_org build_model_cifar10_smallmodel \
  --seeds 42
```

## Run one arm

```bash
python -m examples.phase_logic.run_experiment --experiment pure_ce --parametrization light --seed 42
python -m examples.phase_logic.run_experiment --experiment annealing --parametrization light --seed 42
python -m examples.phase_logic.run_experiment --experiment annealing_phase --parametrization light --seed 42
python -m examples.phase_logic.run_experiment --experiment annealing_phase --model-variant build_model_linear_head --parametrization light --seed 42
```

Useful overrides include:

```bash
python -m examples.phase_logic.compare \
  --seeds 42 43 44 \
  --iterations 1000 \
  --eval-frequency 100 \
  --temperature-warmup 200 \
  --end-temperature 0.2 \
  --phase-margin 1.0 \
  --phase-warmup 200 \
  --phase-lambda-max 0.2 \
  --device cuda
```

Use `--device cpu` if CUDA is unavailable. `--device auto` is the default.

## Results

Each run writes these files beneath
`examples/phase_logic/results/<dataset>/<encoding>/<parametrization>/<model_variant>/<experiment>/seed_<seed>/`
for CIFAR-10 (MNIST retains the previous path layout):

- `config.json`: complete run configuration and resolved device
- `model_report.json`: input thresholds/shapes, parameter count, and
  logic-layer width report
- `metrics.csv`: metrics at every evaluation step
- `summary.json`: final and best discrete results
- `final_model.pt`: only when `--save-model` is passed to the single-run command

The comparison command additionally creates:

- `comparison.csv`: final metrics for every arm and seed
- `paired_differences.csv`: per-seed effect of adding phase regularization
- `head_differences.csv`: per-seed linear-head minus GroupSum differences

For logic convolution layers, `width` is the number of binary output wires
across all spatial positions. The model report also records the number of
spatially shared LUT nodes and the total LUT-node applications. For
`LogicDense`, `width`, output wires, and LUT nodes are all the layer's
`out_dim`.

The primary comparison is the paired change in final discrete accuracy. The
training and test relaxed accuracies show whether the relaxed-accuracy ceiling
comes mainly from optimization/capacity or generalization. The
relaxed-minus-discrete gap, phase energy, and the fraction of logits outside
the margin help explain why accuracy changed.

# my progress: 

-now its works well on mnist dataset . and in 4K better than other in MNIST . 
  model: build_model_linear_head
  experiment: annealing
  seed: 200
  relaxed test: 96.39%
  discrete test: 96.24%

-we could beat BitLogic best-of-space in 4K mode and fan-in 4 by 42.5%(best bitlogic 38%):
  python -m examples.phase_logic.run_experiment --dataset cifar10 --experiment annealing --model-variant build_model_cifar10_org --parametrization light --seed 21 --device cuda --iterations 2500
  [cifar10/annealing/light] model=build_model_cifar10_org seed=21 device=cuda parameters=517120

  now we are testing 16K with fan-in 4 but accuracy is 40%!!!!
  why model capacity extension worsed?
  native WARP[smallconv] works btter on cifar-10 rather than cifar10_org

- model itself could reach to 54% acc in rank=2 on smallmodel on cifar-10 in Light param.

- Rank 4: 1.72M parameters, best discrete 57.89% : with this command
- [best] Rank 2: 59.5% smallmodel+hybrid encoder(quantile+sin)

python -m examples.phase_logic.run_experiment \
  --dataset cifar10 \
  --model-variant build_model_cifar10_smallmodel \
  --input-encoding quantile \
  --num-input-bits 6 \
  --experiment annealing \
  --parametrization light \
  --device cuda \
  --batch-size 8 \
  --iterations 27000

For optional CIFAR-10 training augmentation, add `--cifar-augmentation`.
This applies a reflection-padded random crop and random horizontal flip only
to optimization batches; train-evaluation and test images remain unchanged.
Use `--no-cifar-augmentation` (or omit the option) to disable it.

For cosine learning-rate decay from the initial learning rate to `0.002`, add
`--lr-schedule cosine`. Override the endpoint with `--end-learning-rate`.
The default final LUT temperature is `0.30`; use `--end-temperature` to change it.

python -m examples.phase_logic.run_experiment \
  --dataset cifar10 \
  --model-variant build_model_cifar10_smallmodel \
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
  --iterations 47000


  ### Short Progress Report 8/31/2026

We started with a **quantile encoder + CIFAR10 small logic model** that worked reasonably well, while the original hybrid quantile+sine setup underperformed.

The main encoder issue was the sinusoidal branch. The original sine formulation used `sin(weight*x + phase)` with small effective phase variation over pixel values. we updated `HybridFeatureEncoder` to use stronger frequency scaling with \(2\pi\), minimum sine-frequency protection, and cleaner phase handling. The updated hybrid configuration uses **Q2 + S2**, producing **12 encoded channels**. 

With the updated encoder, the non-augmented hybrid model improved to about **58.9% final discrete accuracy**, showing that the new sine encoding was beneficial.

We then added **CIFAR-10 random crop + horizontal flip augmentation**, while keeping training evaluation and test evaluation clean. This improved generalization substantially. In the 47k augmented run, relaxed test accuracy reached roughly **64%**, compared with around 60% before augmentation. 

The augmented model achieved:

* **334,208 parameters**
* **59.65% final discrete test accuracy**
* **60.97% best observed discrete checkpoint**
* **64.02% final relaxed test accuracy** 

The remaining issue is mainly the **relaxed-to-discrete gap**, rather than generalization itself. we also tested `annealing_phase`, but it did not improve the result, so the current best stable setup is:

**CLGN-Small + Hybrid Q2/S2 + improved sine encoder + crop/flip augmentation + standard annealing.**

Overall, the progression was approximately:

**old hybrid → improved encoder → better hybrid accuracy → augmentation → better generalization and ~60% discrete CIFAR-10 accuracy.**

i changed this one as last edit: k_num 32 was good
encoded_model_type = type(
        "EncodedClgnCifar10Small",
        (ClgnCifar10Small,),
        {"n_input_bits": encoder.num_bits,
        "k_num": 48,}, # was 32
- new command: 
  python -m examples.phase_logic.run_experiment \
  --dataset cifar10 \
  --model-variant build_model_cifar10_smallmodel \
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
  --iterations 57000 \
  --lr-schedule cosine \
  --learning-rate 0.02 \
  --end-learning-rate 0.002 \
  --end-temperature 0.30

  ---- new expriment: logic dense topology:
      ModifiedLogicDense 
      -Train relaxed: up to 73.14%
      -Test relaxed: up to 67.43%
      -Best discrete: 64.72% at step 39k

   -**Best result so far:** **65.87% discrete test accuracy**.

      Best setting:

      * Model: `build_model_cifar10_small_classrouted`
      * Encoder: **Q4 + S2**
      * Topology: **100% class-routed Dense**
      * Params: **375,984**
      * Seed: `4242`
      * Training: `annealing`/pure_ce
      * CIFAR augmentation: enabled

Best checkpoint: about **58k steps**, with relaxed test accuracy around **67.7%**.
    

      (logc_env) spectralm01:~/project/Phasefield_DLGN/torchlogix> python -m examples.phase_logic.run_experiment --dataset cifar10 --model-variant build_model_cifar10_small_classrouted --input-encoding hybrid --num-quantile-bits 4 --num-sine-bits 4 --sine-sigma 1.0 --sine-seed 0 --cifar-augmentation --experiment annealing --parametrization light --seed 4242 --device cuda --iterations 187000 --end-temperature 0.3 --batch-size 128


      python -m examples.phase_logic.run_experiment --dataset cifar100 --model-variant build_model_cifar100_small_classrouted --input-encoding hybrid --num-quantile-bits 4 --num-sine-bits 4 --sine-sigma 1.0 --sine-seed 0 --cifar-augmentation --experiment annealing --parametrization light --seed 4242 --device cuda --iterations 187000 --end-temperature 0.3 --batch-size 128 --cifar-random-erasing --save-model


      
      (logc_env) spectralm01:~/project/Phasefield_DLGN/torchlogix> python -m examples.phase_logic.rr10 --model-variant build_model_cifar10_small_classrouted --input-encoding hybrid --num-quant --experiment annealing --parametrization light --seed 42421239 --device cuda --iterations 18lr-schedule constant --end-temperature 0.2 --batch-size 128 --cifar-augmentation --save-model

      --model-variant build_model_cifar100_small_classrouted_channelmixed
      --model-variant build_model_cifar10_small_classrouted_channelmixed