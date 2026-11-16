# Continuous Adaptive Infusion (CAI v1)

This is a separate candidate-resampling experiment for the repository's CompactLogic models. It trains a **fresh model**, using a run's `config.json` as the starting configuration. It does not load the reference checkpoint or change the existing infusion implementation.

Run from the repository root. The default reference is the best plain CIFAR-10 92,400 × 92,400 run:

```bash
python -m examples.phase_logic.infusion_training.new_continous_infusion.train --dry-run
python -m examples.phase_logic.infusion_training.new_continous_infusion.train
```

Choose a different CompactLogic model deliberately with `--model-variant`. For a different dataset, provide a matching run configuration with `--reference-config` and optionally override its model:

```bash
python -m examples.phase_logic.infusion_training.new_continous_infusion.train \
  --model-variant build_model_cifar10_small_compact_head \
  --compact-freeze-mode none --dry-run

python -m examples.phase_logic.infusion_training.new_continous_infusion.train \
  --reference-config path/to/cifar100/run/config.json \
  --model-variant build_model_cifar100_small_compact \
  --output-dir examples/phase_logic/infusion_training/new_continous_infusion/results/my_cifar100_run \
  --dry-run
```

Remove `--dry-run` to train. Available overrides include `--dataset`, `--model-variant`, input encoding and bit settings, augmentation, batch size, learning rate, workers, seed, device, backend, iterations, exploration steps, evaluation frequency, and `--compact-freeze-mode {none,entropy,hybrid}`. The dataset, model, and encoding must be compatible under `ExperimentConfig.validate()`. Using a reference config from the same dataset is simplest. Each output directory must be new.

Use `--infusion-profile strong_unary` to opt into a stronger unary candidate prior. The default profile is unchanged. At α=1, `strong_unary` draws Dense1 candidates with 10% uniform, 70% identity, 15% inversion, and 5% constant; Dense2 uses 20% uniform, 60% identity, 15% inversion, and 5% constant. Intermediate α values interpolate from uniform sampling exactly as in the default profile. Convolution preferences and CAI feedback, rollback, and freezing are unchanged. When `--output-dir` is omitted, the strong profile gets a `_strong_unary` suffix to keep its results separate.

The default `--step-down-mode fractional` preserves the original time-scaled response. Use `--step-down-mode absolute --step-down 0.05` to subtract exactly 0.05 from the controller target on each decrease decision; α is still smoothed toward that target.

The policy finds candidate layers in the selected model. It biases every convolution except the final one toward AND/OR; the final convolution stays uniform. It applies the Dense1 mixture to the first dense stage and the Dense2 mixture to subsequent dense stages. All engines in a class-routed dense stage receive the same mixture. At `alpha=1`, Conv1–3 use 75% uniform plus 25% AND/OR and Conv4 stays fully uniform. Dense1 uses 10% uniform, 50% identity, 10% inversion, and 30% constants; Dense2 uses 20% uniform, 45% identity, 10% inversion, and 25% constants. Uniform draws can still select all 16 operations. On models with extra convolution branches, this traversal-order policy is experimental and does not imply the same branch-specific semantics as the reference model.

CAI starts at `alpha=0.7`. At each validation evaluation, it compares a time-scaled accuracy EMA with its best value. Healthy accuracy moves α smoothly toward 1; a sustained drop beyond `--accuracy-tolerance` moves it toward 0. `--step-up` and `--step-down` are fractions of the remaining distance per 5% of exploration, rather than fixed α increments. The response accounts for elapsed training steps, so changing evaluation frequency does not inherently slow adaptation. There is no cooldown gate. Rollback requires three consecutive validation results below the safe accuracy by more than `--rollback-drop`; any acceptable result resets this confirmation streak. The model, optimizer, controller, and RNG snapshot are restored only after the third failure. Test accuracy does not control these decisions. The 90/10 training/validation split matches progressive infusion. Rollback does not rewind the active data-loader iterator, so this entry point does not claim bit-exact resume.

Outputs include `config.json`, `model_report.json`, `metrics.csv`, `summary.json`, `best_model.pt`, and a safe snapshot when one is created. Hybrid freezing also records `freeze_probes.csv`.

At each validation checkpoint, `metrics.csv` records cumulative weak/strong candidate-resampling counts separately from actual changes in the selected Boolean operations. It includes the selected-gate change count/rate, the change rate among currently non-frozen gates, two-input-to-simple and simple-to-two-input transition rates, and the current unary/two-input/constant composition. The same fields are emitted with `conv1_` … `conv4_` and `dense1_` … `dense2_` prefixes for the reference architecture. Class-routed engines are combined into their logical dense stage. Rates use all gates in the reported scope as their denominator, except `eligible_gate_change_rate`, whose denominator is currently non-frozen gates. The first checkpoint establishes the baseline and therefore reports zero transitions.

After a completed run, `gate_dynamics.png` plots selected-gate change rate, simplification/complexification rates, and α versus training step. Tracking reads one selected operation byte per output gate only at validation checkpoints; it does not alter weights, candidates, RNG state, CAI feedback, or resampling. CAI does not calibrate its initial strength; changes to α affect only future resampling events and do not undo already selected or frozen gates. If no candidate rows are resampled, changing α has no immediate effect on the network.



# commands:
python -m examples.phase_logic.infusion_training.new_continous_infusion.train \
  --model-variant build_model_cifar10_small_compact_head \
  --iterations 1150000 \
  --exploration-steps 600000 \
  --eval-frequency 50000 \
  --seed 424244921 \
  --batch-size 128 \
  --learning-rate 0.02 \
  --num-workers 4 \
  --device cuda \
  --compact-backend cuda \
  --compact-freeze-mode hybrid \
  --initial-alpha 0.95 \
  --infusion-profile strong_unary \
  --output-dir examples/phase_logic/infusion_training/new_continous_infusion/results/aggressive_cifar10_compact_head_cai_strong_unary_seed424244921_al95_eval50000


# new model no skip update:
python -m examples.phase_logic.infusion_training.new_continous_infusion.train \
  --model-variant build_model_cifar10_small_compact \
  --iterations 1150000 --exploration-steps 600000 --eval-frequency 50000 \
  --seed 424244921 --batch-size 128 --learning-rate 0.02 --num-workers 4 \
  --device cuda --compact-backend cuda --compact-freeze-mode hybrid \
  --initial-alpha 0.95 --step-down 0.05 --step-down-mode absolute \
  --infusion-profile strong_unary \
  --output-dir examples/phase_logic/infusion_training/new_continous_infusion/results/aggressive_cifar10_compact_plain_cai_strong_unary_seed424244921_al95_absolute_down005_eval50000