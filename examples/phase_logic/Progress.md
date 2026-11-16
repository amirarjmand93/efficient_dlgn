## DLGN Progress and Remaining Challenges

The current DLGN has improved through several targeted changes while preserving its rank-2 Boolean/LUT structure and hardware-friendly design. The most successful improvements so far have been the **multi-bit input encoding**, especially quantile-based and hybrid Q+S representations, and **CIFAR random crop + horizontal-flip augmentation**, which produced a clear accuracy improvement. Class-routed dense layers were also explored, although their benefit appears smaller and less consistent.

### Additional dataset support for the plain 92.4k CompactLogic model

Added `build_model_cifar100_small_compact` and `build_model_mnist_small_compact`
using the same 92,400-by-92,400 shared dense head and no-skip four-convolution
backbone. CIFAR-100 uses the existing RGB Q2+LBP encoder and `GroupSum(k=100)`;
MNIST uses the existing binarized grayscale 28x28 loader and `GroupSum(k=10)`.
Both variants accept progressive aggressive infusion and hybrid convolution
freezing controls. Commands are in `README.md`. These variants are registered
but have not been trained, so no accuracy comparison is available.

Example CIFAR-100 training command:

```bash
python -m examples.phase_logic.run_experiment \
  --dataset cifar100 \
  --model-variant build_model_cifar100_small_compact \
  --input-encoding hybrid --hybrid-features lbp --num-quantile-bits 2 \
  --cifar-augmentation \
  --experiment pure_ce --parametrization compact --compact-backend cuda \
  --compact-exploration-steps 800000 --iterations 1500000 \
  --eval-frequency 15000 --learning-rate 0.02 --batch-size 128 \
  --num-workers 4 --seed 4242449222 --device cuda --save-model \
  --compact-progressive-aggressive-infusion --compact-freeze-mode hybrid \
  --compact-freeze-max-taylor 0.2 \
  --compact-freeze-max-loss-increase 0.099 \
  --compact-freeze-max-probe-drift 1.0
```

Example MNIST training command for the matching compact topology:

```bash
python -m examples.phase_logic.run_experiment \
  --dataset mnist \
  --model-variant build_model_mnist_small_compact \
  --input-encoding fixed --num-input-bits 1 \
  --experiment pure_ce --parametrization compact --compact-backend cuda \
  --compact-exploration-steps 800000 --iterations 1500000 \
  --eval-frequency 15000 --learning-rate 0.02 --batch-size 128 \
  --num-workers 4 --seed 4242449222 --device cuda --save-model \
  --compact-progressive-aggressive-infusion --compact-freeze-mode hybrid \
  --compact-freeze-max-taylor 0.2 \
  --compact-freeze-max-loss-increase 0.099 \
  --compact-freeze-max-probe-drift 1.0
```

The network still follows the same core pipeline:

**binary encoder → LogicConv layers → LogicDense/class-routed layers → GroupSum classification**

Several alternatives—including attention, extra convolutional depth, larger widths, random wire erasure, random erasing, and modified dense structures—did not provide reliable gains.

### LogicConv remains an important research direction

The **LogicConv layers are not fully solved and may still be a major bottleneck**. In the current implementation, every logic tree uses a small fixed subset of input wires. In the first convolution, this can strongly restrict interaction between different encoded channels such as RGB, quantile, and sine bits.

Simply increasing the channel group size did not improve performance, and the first `channel-balanced` experiment also appeared too aggressive: mixing many channels within only eight available tree leaves reduces the amount of spatial information seen from each channel.

A more promising direction is therefore to redesign LogicConv connectivity so that each kernel jointly preserves:

- **spatial structure** inside the receptive field,
- **cross-channel interaction** between encoded features,
- **balanced channel usage** across kernels,
- and the same small rank-2 logic-tree/hardware cost.

For example, a kernel could use **two complementary channels across four shared spatial locations**, rather than eight unrelated channels from eight individual pixels.

### Main remaining challenges

The highest-priority areas are therefore:

1. **LogicConv connectivity** — improve channel mixing without destroying spatial structure.
2. **Generalization / overfitting** — reduce the gap between training and test performance.
3. **Fixed topology** — investigate whether better or partially learned/stochastic connectivity can outperform one randomly fixed wiring.
4. **Relaxed-to-discrete gap** — improve conversion from the differentiable training model to the final Boolean network.
5. **Dense head efficiency** — determine whether the large LogicDense section can be simplified or regularized without reducing accuracy.

The most promising current direction is therefore not simply making the DLGN wider or deeper, but improving **how information is connected and composed inside the existing logic gates** while keeping the final network compact and hardware-friendly.

# Model Cifar10 Variants:
  --model-variant build_model_cifar10_small_classrouted_compact_head
  --model-variant build_model_cifar10_small_compact [no skip connection-92.4k x 92.4k]
  --model-variant build_model_cifar10_small_compact_head
  --model-variant build_model_cifar10_compact_small [no skip connection-80*80 orginal size]
  --model-variant build_model_cifar10_small_compact_head_wide75

  # Skip-connection ablation [adopted]
  The plain `build_model_cifar10_small_compact` run with 92,400 x 92,400 dense
  layers and no concatenation skip connections reached 70.72% peak discrete
  test accuracy. This is comparable to the skip-connected model, so removing
  the skip connection did not reduce accuracy in this experiment. We no longer
  use skip connections in the selected architecture; this also avoids their
  additional long-range routing connections.

  Evidence:
  `results/cifar10/hybrid_q2_lbp2/crop_flip_augmentation/compact_shared_head2_v3_plain_w92400x92400_cuda_c16_explore800000_resample100_freeze200`

python -m pip install --no-build-isolation '.[compact]'

python -m examples.phase_logic.run_experiment \
  --dataset cifar10 \
  --model-variant build_model_cifar10_compact_small \
  --parametrization compact --experiment pure_ce \
  --input-encoding thermometer --num-input-bits 3 \
  --cifar-augmentation --compact-backend cuda \
  --iterations 600000 --eval-frequency 2000 \
  --learning-rate 0.02 --batch-size 128 \
  --num-workers 4 --seed 0 --device cuda --save-model
  

  it was great ! 66% accuracy . now i am testing my model+ compact logic learning method :

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
  --learning-rate 0.02 --batch-size 128 \
  --num-workers 4 --seed 42421015 --device cuda --save-model

  # our new flags :
   --model-variant build_model_cifar10_compact_sixconv 
   --compact-freeze-mode hybrid

   # lets test cifar-100 on previous code we had [torchlogix buildup]

   # Testing less restrictive hybrid freezing :
   python -m examples.phase_logic.run_experiment \
  --dataset cifar10 \
  --model-variant build_model_cifar10_small_classrouted_compact_head \
  --input-encoding hybrid --num-quantile-bits 2 --num-sine-bits 2 \
  --sine-sigma 1.0 --sine-seed 0 --cifar-augmentation \
  --experiment pure_ce --parametrization compact --compact-backend cuda \
  --compact-exploration-steps 50000 --iterations 70000 \
  --eval-frequency 2000 --learning-rate 0.02 --batch-size 128 \
  --num-workers 4 --seed 4242449 --device cuda --save-model \
  --compact-freeze-mode hybrid \
  --compact-freeze-max-taylor 0.20 \
  --compact-freeze-max-loss-increase 0.08

  --compact-freeze-max-probe-drift 1.0

  # Best CLI - no skip-connection
  # new flag for Quantile + Binary Spatial Direction Encoding (BSDE):
  --input-encoding hybrid --hybrid-features lbp --num-quantile-bits 2

  python -m examples.phase_logic.run_experiment \
  --dataset cifar10 \
  --model-variant build_model_cifar10_small_compact \
  --input-encoding hybrid --hybrid-features lbp --num-quantile-bits 2 \
  --cifar-augmentation \
  --experiment pure_ce --parametrization compact --compact-backend cuda \
  --compact-exploration-steps 600000 --iterations 1200000 \
  --eval-frequency 15000 --learning-rate 0.02 --batch-size 128 \
  --num-workers 4 --seed 4242449 --device cuda --save-model \
  --compact-freeze-mode hybrid \
  --compact-freeze-max-taylor 0.20 \
  --compact-freeze-max-loss-increase 0.08 \
  --compact-freeze-max-probe-drift 1.0 \
  --compact-efficient-layer-aware-resampling

  # flga with no freezing :
  --compact-freeze-mode hybrid --compact-freeze-min-interval 1000000

  it seems adding 10% dense is more effective in acc rather than 10% in conv layer

  # new Implemented[Failed] the separate MRLB-v2 model[ a bridge in conv layes]. Baseline unchanged.
    Use your existing CLI with:
    --model-variant build_model_cifar10_small_mrlb_compact_head \
    --mrlb-conv3-relational-channels 128

  # use Gradiant vs minibatch vs iter[Future]
    My recommendation: use gradient variance respet to dense layers
    Make two figures for your paper:
    1. 3D gradient magnitude: Iteration × mini-batch[quantile] × gradient norm.
    2. 2D normalized gradient variance: Iteration × gradient variance, comparing freezing versus no freezing.  

  # hybrid freezing desnse layer [Failed]
    --dense-freeze-fraction 0.10 
    After all convs freeze, it freezes Dense1 → Dense2 → Dense3, in 10% groups.

  # NO Layers freezing 
    --compact-freeze-mode none  

  # 3D histogram of gates/layer/frequncy [deploy]

    python -m examples.phase_logic.plot_gate_histogram \
    --checkpoint "examples/phase_logic/results/cifar10/hybrid_q2_lbp2/crop_flip_augmentation/compact_shared_head_v7_concat_c128x160x512x512_w88000x88000_cuda_c16_explore500000_resample100_freeze200/hybrid_freeze_v3_probedriven_n4_sigma2_gap5000_probe1000_confirm3_drift1_flip0.001_taylor0.2_loss0.08/compact/build_model_cifar10_small_classrouted_compact_head/pure_ce/seed_4242449/best_model.pt" \
    --output examples/phase_logic/plots/shared_head_v7_widened_conv_gates_3d.png \
    --frequency percent  

  # new resampling gate selection or gate prefrence [deploy]  
    
    --compact-layer-aware-resampling
    add optional layer-aware candidate resampling, keeping the baseline unchanged.
    Conv1–3: 25% AND/OR preference + 75% uniform.
    Dense2: 25% XOR/XNOR preference + 75% uniform.
    Conv4/Dense1 unchanged.

    
    --compact-efficient-layer-aware-resampling:[good]
    update ELAR with a new %:
  - Dense1: 70% uniform, 18% A/B, 8% NOT A/B, 4% TRUE/FALSE.
  - Dense2: 80% uniform, 12% A/B, 6% NOT A/B, 2% TRUE/FALSE.
  - Conv1–3: Keep 75% uniform + 25% AND/OR.
  - Conv4: Keep 100% uniform.


  --compact-aggressive-layer-aware-resampling

    | Layer | Uniform | A/B | NOT A/B | TRUE/FALSE |
    |---|---:|---:|---:|---:|
    | Dense1 | 50% | 30% | 5% | 15% |
    | Dense2 | 60% | 25% | 5% | 10% |

    

    --compact-progressive-aggressive-infusion
      5 level
      Changed the Progressive Aggressive Infusion defaults:
    - Rollback accuracy drop: 1.5 pp (0.015)
    - Rollback loss increase: 0.10
    - Stable accuracy tolerance: 0.3 pp (0.003)
    - Stable loss tolerance remains 0.05
    - Cooldown remains 15,000 steps
    - Stability requirement remains 3 evaluations

    Result [accuracy regression / not selected]:
    The controller reached Level 5 at step 525,000, where candidate infusion
    strongly preferred identity and constant gates. The run achieved 69.38%
    peak discrete test accuracy and finished at 69.19%, below the 70.72% result
    recorded for the selected plain no-skip model. This experiment indicates
    that progressively increasing identity/TRUE/FALSE pressure reduced model
    accuracy. We therefore do not use progressive aggressive infusion in the
    selected accuracy-oriented configuration. The comparison is directional,
    since the two runs also differ in backbone/skip configuration, seed, and
    exploration length.

    Evidence:
    `results/cifar10/hybrid_q2_lbp2/crop_flip_augmentation/compact_shared_head2_v2_concat_w92400x92400_cuda_c16_explore700000_resample100_freeze200/progressive_aggressive_infusion_v3_levels5_stable3_cooldown15000_rollback1p5_val10`

    --compact-progressive-aggressive-infusion-unary-biased [new, untested]
    Alternative five-level dense candidate schedule; use this flag instead of
    `--compact-progressive-aggressive-infusion`. Constant preference stays at
    15%, while identity and inversion preferences rise across levels. Conv1–3
    keep the existing 25% AND/OR preference, with the same validation-based
    advancement and rollback rules.

    | Level | Uniform | Identity A/B | NOT A/B | TRUE/FALSE |
    |---|---:|---:|---:|---:|
    | 1 | 50% | 30% | 5% | 15% |
    | 2 | 45% | 30% | 10% | 15% |
    | 3 | 40% | 35% | 10% | 15% |
    | 4 | 35% | 35% | 15% | 15% |
    | 5 | 30% | 40% | 15% | 15% |



  # my ablation study :
    My preferred figure
    Plot the normalized gradient variance for Dense1 and Dense2, comparing:
    - Original training without freezing
    - Hybrid progressive freezing
    - ELAR + hybrid freezing (if evaluating the combined method)

  # vivado command: 
    cd examples/phase_logic/post_training/outputs/best_shared_head_92400/circuit/run_synthesis
    
    //run 3 test
    vivado -mode batch -source synthesize.tcl \
      -tclargs ../circuit.v compare fpga_reports 10.0  

    // run artix7 test
   vivado -mode batch -source synthesize.tcl \
   -tclargs ../circuit.v xc7a35tcpg236-1 artix_reports 10.0
