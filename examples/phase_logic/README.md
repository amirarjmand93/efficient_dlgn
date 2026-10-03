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

The encoded channel count is derived from the selected encoding settings and
passed to the model through `encoder.output_channels`. For example:

- Q2 + S2 produces 12 channels for RGB images.
- Q4 produces 12 channels.
- Q4 + S4 produces 24 channels.

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
- `build_model_cifar10_small_modifieddense`
- `build_model_cifar10_small_classrouted`
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

The double-well phase term has stable points at `-margin` and `+margin`.

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
- `final_model.pt`: final checkpoint when `--save-final-model` is enabled.

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
