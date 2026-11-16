# Five-class heartbeat experiment

Use `data/heartbeat/mitbih_train.csv` and `mitbih_test.csv` from the Kaggle
heartbeat dataset. Each row contains 187 normalized waveform samples followed
by a class label from 0 to 4. The PTB-DB CSVs are a separate binary task and
are not used here.

`build_model_heartbeat_small_compact` keeps the plain CompactLogic CIFAR model's
four convolution widths (128, 128, 512, 512), two fully shared dense widths
(92,400, 92,400), 16 candidates per gate, and `GroupSum(tau=55)`. The output
uses `GroupSum(k=5)`. The new fixed encoder uses two quantile thresholds from
the training split and two adjacent-sample comparison bits. It then reshapes
the encoded 187-sample sequence to 11 x 17 for the existing 2D convolutions.
This changes convolutional feature-map sizes and inference node count, so
hardware costs are not directly matched to 32 x 32 CIFAR images.

Run from the repository root:

```bash
python -m examples.phase_logic.run_experiment \
  --dataset heartbeat \
  --model-variant build_model_heartbeat_small_compact \
  --input-encoding hybrid --hybrid-features lbp --num-quantile-bits 2 \
  --experiment pure_ce --parametrization compact --compact-backend cuda \
  --compact-exploration-steps 800000 --iterations 1500000 \
  --eval-frequency 15000 --learning-rate 0.02 --batch-size 128 \
  --num-workers 4 --seed 4242449222 --device cuda --save-model \
  --compact-freeze-mode hybrid \
  --compact-freeze-max-taylor 0.20 \
  --compact-freeze-max-loss-increase 0.099 \
  --compact-freeze-max-probe-drift 1.0
```

The current training report uses overall accuracy. MIT-BIH classes are
imbalanced; inspect per-class recall or macro F1 before comparing methods.
The provided beat-level split also does not establish patient-independent
generalization.
