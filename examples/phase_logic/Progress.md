## DLGN Progress and Remaining Challenges

The current DLGN has improved through several targeted changes while preserving its rank-2 Boolean/LUT structure and hardware-friendly design. The most successful improvements so far have been the **multi-bit input encoding**, especially quantile-based and hybrid Q+S representations, and **CIFAR random crop + horizontal-flip augmentation**, which produced a clear accuracy improvement. Class-routed dense layers were also explored, although their benefit appears smaller and less consistent.

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

# Model Variants:
  --model-variant build_model_cifar10_small_classrouted_compact_head
  --model-variant build_model_cifar10_small_compact_head
  --model-variant build_model_cifar10_small_compact_head_wide75

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

  # Best CLI
  # new flag for Quantile + Binary Spatial Direction Encoding (BSDE):
  --input-encoding hybrid --hybrid-features lbp --num-quantile-bits 2

  python -m examples.phase_logic.run_experiment \
  --dataset cifar10 \
  --model-variant build_model_cifar10_small_compact_head \
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