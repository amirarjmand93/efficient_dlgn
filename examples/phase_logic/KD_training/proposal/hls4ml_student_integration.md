# Proposal: hls4ml Integration for the LUT6 Student

## Decision

Integrate the trained student with `hls4ml` through a small custom LUT6 extension. Do **not** make TorchScript or `torch.fx` the primary conversion path. The student contains custom Boolean lookup, input-selection, convolution-patch and GroupSum operations that the standard PyTorch frontend does not understand, and its Python forward path contains tracing-hostile control flow.

The conversion boundary should instead be the already-discrete model representation:

```text
best_student.pt
  -> instantiate the matching student and load student_state
  -> model.eval() and model.discrete_state()
  -> validated discrete LUT6 representation
  -> custom hls4ml LUT6 layers
  -> HLS C++
  -> C simulation / synthesis / RTL co-simulation
```

This is a narrow `torchlogix LUT6 -> hls4ml` adapter, not a general PyTorch compiler.

## Why this revision is preferable

The checkpoint normally stores a `state_dict`, configuration and validation metadata rather than a self-contained computational graph. Reconstructing the model from `student.py` is therefore still required, but the live PyTorch model should only be used to recover and verify its discrete state. Asking the normal hls4ml PyTorch frontend to infer these custom semantics would be fragile.

The repository already has most of the trustworthy front half:

- `student.py` defines `TinyConvLUT6Student` and `TinyConv2DenseLUT6Student` and exposes `discrete_state()`.
- `post_training/load_checkpoint.py` validates the two-dense discrete architecture and checkpoint contents.
- `post_training/student_circuit_exporter.py` expands the model into exact LUT6 connectivity and verifies integer class counts.
- `torchlogix.circuit.NativeLUT6Circuit` provides a simplified, serializable circuit representation with constant offsets preserved in GroupSum.

The recommended hls4ml importer should consume a validated discrete layer representation. `NativeLUT6Circuit` can remain the equivalence oracle and may be used as the serialized input later, but the first implementation should retain layer geometry rather than importing only a fully flattened gate graph. Geometry makes convolution scheduling and resource reuse much easier.

## Initial model scope

Begin with `tinyconv_q2_lbp2`, the smaller one-dense model:

```text
12 x 32 x 32 encoded Boolean input
  -> LUT6Conv2D: 12 to 32 channels, 3x3, stride 2, padding 1
  -> LUT6Conv2D: 32 to 64 channels, 3x3, stride 2, padding 1
  -> LUT6Dense: 4096 to 8000 Boolean outputs
  -> GroupSum: 10 groups of 800
  -> 10 unsigned integer class counts
```

After this works, add `tinyconv_q2_lbp2_dense2` by inserting the second `8000 -> 8000` LUT6 layer. The current post-training loader supports only this two-dense variant, so a future implementation must generalize it cleanly rather than pretending that it can already load the one-dense checkpoint.

Keep the Q2+LBP2 RGB encoder outside the first HLS core. The initial input is the existing `12 x 32 x 32` Boolean tensor in CHW order. Moving the encoder into HLS is a separate second-stage task.

## Required custom hls4ml layers

### 1. LUT6Conv2D

Static configuration:

- input/output shapes;
- six selected patch indices for every output channel;
- one 64-bit truth table for every output channel;
- kernel size 3, stride 2, padding 1 and zero padding;
- address convention: pin 0 is the least-significant address bit.

The HLS implementation forms a six-bit address and reads one truth-table bit. The same selected inputs and truth table are shared across spatial positions for a channel, matching the trained model.

### 2. LUT6Dense

Static configuration:

- six source indices per output node;
- one 64-bit truth table per output node;
- fixed input and output widths.

This layer must preserve every selected connection exactly. Resource sharing may change the evaluation schedule, but not the graph.

### 3. LUT6GroupSum

Static configuration:

- ten ordered groups;
- 800 logical outputs per class before simplification;
- any constant offsets produced by circuit simplification;
- sufficient unsigned result width (10 bits is enough for 0--800, although the existing exporter uses 16-bit score ports).

For a parallel implementation, use balanced partial popcounts and a balanced addition tree. For the preferred resource-shared implementation, accumulate counts over multiple cycles with registers. Both must produce the exact same integer class counts.

## hls4ml integration components

The future implementation should contain only these pieces:

1. **Checkpoint/discrete-state adapter** -- reconstruct the selected student class, load the state, call `eval()` and extract fixed connections and Boolean truth tables.
2. **Validated layer IR** -- preserve layer geometry, shapes, flattening order, truth-table address convention, GroupSum membership, scale metadata and folded constants.
3. **hls4ml layer definitions** -- `LUT6Conv2D`, `LUT6Dense` and `LUT6GroupSum` layer objects with shape inference.
4. **Backend templates** -- HLS C++ declarations, configuration structures and function calls for the selected backend.
5. **Registration entry point** -- one explicit function that registers the custom layers, parsers and templates with hls4ml. Avoid import-time global side effects.
6. **Converter API** -- accept a checkpoint or validated IR and return an hls4ml model/configuration. Keep it callable from other Python scripts.

No generic code emitter, graph optimizer or new training framework is needed.

## Hardware architecture recommendation

The main reason to use hls4ml here is to explore a **clocked, multicycle, resource-shared** implementation. Fully unrolling all LUT6 evaluations would largely recreate the combinational RTL and its routing-congestion risk.

Expose a small number of scheduling controls:

- LUT engines per layer, or an equivalent reuse factor;
- truth-table storage implementation;
- interface type and target clock period;
- optional dataflow between layers after the basic sequential design is correct.

Start conservatively with one image processed at a time and modest parallelism. Store intermediate Boolean feature maps in local arrays or memories, evaluate nodes in regular loops, and accumulate GroupSum locally. Increase parallelism only after synthesis reports establish the resource/latency trade-off.

Do not require bit-accurate output timing to match the combinational exporter. Functional equivalence means identical ten integer class scores for each input; latency may differ.

## Numeric and semantic contract

The converter must preserve all of the following:

- Boolean input and intermediate values;
- the exact six selected connections for every LUT;
- all 64 truth-table bits;
- pin 0 as the LUT address LSB;
- zero padding and patch order `channel,row,column`;
- dense flattening order `channel,row,column`;
- class grouping and any folded constant offsets;
- raw integer counts as the hardware output.

`student_tau` is not needed to choose the winning class because it is a common positive divisor. Keep it as metadata. If floating-point logits are ever exposed, apply `count / student_tau` outside the Boolean HLS core.

## Verification plan and acceptance criteria

Verification should be incremental:

1. Compare the extracted discrete state with the hard-forward PyTorch model.
2. Compare a Python evaluator of the layer IR with `NativeLUT6Circuit`.
3. Run hls4ml C simulation on saved verification vectors.
4. Require exact equality of all ten integer class counts, not only equal `argmax`.
5. Run HLS synthesis and record LUT, FF, BRAM/URAM, latency, initiation interval and estimated clock.
6. Run RTL co-simulation on the same vectors.
7. Only then run Vivado implementation and record routing success and post-route timing.

Test all-zero and all-one encoded inputs, random Boolean inputs, and encoded CIFAR-10 samples. The final functional acceptance condition is zero score mismatch across all tested vectors. The FPGA acceptance condition is successful routing with no unresolved conflicts and timing appropriate for the chosen clock.

## Suggested implementation order

1. Generalize checkpoint loading to support both one-dense and two-dense students.
2. Define and validate the layer-preserving discrete IR.
3. Implement and unit-test a standalone C++ LUT6 lookup primitive.
4. Add `LUT6Dense`, then GroupSum, and verify a small synthetic network.
5. Add `LUT6Conv2D` and verify the one-dense student end to end.
6. Synthesize several reuse settings and select a reasonable resource/latency point.
7. Add the second dense layer and test the two-dense student.
8. Consider moving Q2+LBP2 encoding into HLS only after the LUT core is verified.

## Explicit non-goals for the first version

- Parsing arbitrary PyTorch or TorchScript models.
- Supporting training in hls4ml.
- Reproducing PyTorch surrogate-gradient operations.
- Automatically changing or pruning trained connections.
- Fully unrolling the network by default.
- Supporting every hls4ml backend immediately.
- Integrating raw RGB input before the encoded-input core works.

## Primary risks

- The dense layers still contain irregular memory accesses. Resource sharing lowers simultaneous routing demand but may increase latency substantially.
- Large compile-time truth-table and connection arrays can create long HLS compile times or inefficient ROM inference; synthesis reports must guide storage choices.
- Excessive unrolling can reproduce the original routing problem.
- Simplifying LUTs before import changes the scheduled graph. This is valid only when GroupSum offsets and all remaining connections are preserved and exact equivalence is rechecked.
- hls4ml extension APIs can change between releases, so pin and record a tested hls4ml version when implementation begins.

## Recommendation

Proceed only if a multicycle architecture is acceptable. In that case, the custom extension is realistic and useful: it can trade latency for much lower parallel routing pressure while preserving the trained Boolean function exactly. If a single-cycle, fully combinational circuit is mandatory, the existing native Verilog exporter is the better route and hls4ml adds little value.
