# Post-training deployment of the LUT6 student

This directory contains two deployment paths for the trained two-dense LUT6
student:

1. A native, fully combinational Verilog circuit exporter.
2. A custom hls4ml bridge for generating a clocked HLS C++ project that can be
   explored with AMD Vitis HLS.

Both paths preserve the trained discrete model. They do not retrain the
network, change its learned connections, rewrite its truth tables, or add
trainable parameters.

## Deployed model boundary

The currently supported student is:

```text
Pre-encoded Q2+LBP2 Boolean input: 12 x 32 x 32, CHW
  -> LUT6Conv2D: 32 channels, 3x3, stride 2, padding 1
  -> LUT6Conv2D: 64 channels, 3x3, stride 2, padding 1
  -> flatten: 64 x 8 x 8 = 4,096 bits
  -> LUT6Dense: 4,096 -> 8,000 bits
  -> LUT6Dense: 8,000 -> 8,000 bits
  -> BooleanGroupSum: 10 groups of 800 bits
  -> 10 raw integer class counts
```

Each LUT6 has six fixed input connections and an exact 64-bit truth table. Pin
0 is the least-significant truth-table address bit:

```text
address = b0 + 2*b1 + 4*b2 + 8*b3 + 16*b4 + 32*b5
output  = truth_table[address]
```

The RGB-to-Q2+LBP2 encoder remains outside the generated hardware core. The
hardware input is the resulting 12,288-bit Boolean tensor in CHW order. Each
class output is an unsigned integer count in `[0,800]`; the original training
logit is `count / student_tau`. Because `student_tau` is the same positive
constant for every class, classification can use `argmax(class_count)` without
performing floating-point division in hardware.

## First approach: fully combinational native Verilog

`student_circuit_exporter.py` converts either `best_student.pt` or
`discrete_student.pt` into a native LUT6 circuit. It produces:

- `circuit.v`: Xilinx `LUT6` primitives and hierarchical GroupSum logic;
- `circuit.json`: reloadable, simplified native-LUT6 circuit IR;
- `circuit.c`: portable integer reference implementation;
- `export_report.json`: simplification counts and verification results;
- `verification_vectors.npz`: encoded Boolean inputs and exact class counts.

The shared `NativeLUT6Circuit` simplifier propagates constants and aliases,
cofactors LUT truth tables, deduplicates equivalent LUTs, eliminates dead
nodes, and folds constant Dense2 outputs into exact GroupSum offsets. The
report records gate counts before and after simplification and verifies the raw
IR, simplified IR, reloaded JSON, and optionally compiled C.

```bash
env PYTHONPATH=src:. python -m \
  examples.phase_logic.KD_training.post_training.student_circuit_exporter \
  --checkpoint path/to/best_student.pt \
  --output-dir examples/phase_logic/KD_training/post_training/outputs/my_student/circuit \
  --compile-c \
  --c-opt-level 0
```

The top module is `circuit`. It is fully combinational: it has no clock,
registers, pipeline, RGB interface, or board I/O. The convolution definitions
are shared in the trained model, but a spatially parallel RTL implementation
expands their applications into distinct physical LUTs and wires. Before
simplification, this student performs 28,288 spatial LUT6 evaluations per
image; GroupSum arithmetic is additional.

### What the earlier combinational ELAR experiments showed

The routing results below belong to the larger ELAR network with two
92,400-node dense layers, not to this 8,000-node student. That native RTL
successfully reached synthesis and placement on the VU19P, but its large
irregular Boolean graph remained difficult to route. Hierarchical GroupSum
improved the reduction structure but did not remove the dominant inter-layer
routing demand. A subsequent 204-tile dense-locality experiment reduced
initial-routing node overlaps from 931,154 to 863,958, a 7.22% improvement,
but routing still failed at congestion level 7.

The combined GroupSum/locality version also used 313,555 LUT primitives,
compared with 259,356 for the earlier aggressive RTL synthesis. These counts
are not an isolated cost measurement for locality because the compared RTL
versions include multiple changes, but they show that logical restructuring
alone did not guarantee a physically routable implementation.

The lesson motivating this student deployment path is that reducing one
reduction bottleneck or reorganizing RTL names does not remove a model's full
set of irregular inter-layer wires. In a fully combinational design, all LUT
evaluations and their interconnect are simultaneously present. The smaller
student has not yet been shown to reproduce the larger ELAR routing failure,
but the earlier result motivates evaluating it through a second lowering path
where Vitis HLS can explore clocked scheduling and hardware reuse.

## New approach: custom hls4ml LUT6 bridge

Stock hls4ml does not understand this model's learned LUT6 operators. The new
bridge therefore uses the hls4ml Extension API to introduce three custom
operations:

- `LUT6Conv2D`
- `LUT6Dense`
- `BooleanGroupSum`

The bridge is not a generic PyTorch or ONNX converter. It is a narrow lowering
path for the exact discrete TorchLogix LUT6 representation:

```text
best_student.pt or discrete_student.pt
  -> validated discrete connections and Boolean truth tables
  -> hls4ml_lut6_ir_v1 JSON
  -> custom hls4ml ModelGraph
  -> registered Vitis/Vivado templates and nnet_lut6.h
  -> generated HLS C++ project
  -> C simulation, HLS synthesis, and implementation as separate later steps
```

### Hardware-oriented JSON IR

`student_hls4ml_exporter.py` creates a self-contained, layer-oriented JSON
document. It contains:

- the Boolean input shape, CHW flattening, and Q2+LBP2 encoder metadata;
- explicit graph edges and execution order;
- convolution geometry, padding, local-patch indexing, and weight sharing;
- every learned six-input connection vector;
- every exact 64-bit LUT truth table as `init_hex`;
- Dense layer shapes and connectivity;
- exact class-to-node GroupSum ranges and the original `student_tau`.

The JSON contains all deployed model parameters. The original `.pt` file is
not needed after this IR has been generated.

### Custom hls4ml lowering

`hls4ml_bridge/` registers custom hls4ml `Layer` classes, configuration
templates, function-call templates, and the synthesizable `nnet_lut6.h`
implementation. The generated design uses:

- `ap_uint<1>` for inputs and intermediate Boolean tensors;
- `ap_uint<6>` for LUT addresses;
- compile-time constant connection arrays and `ap_uint<64>` truth tables;
- `ap_uint<10>` for the ten GroupSum class counts;
- exact CHW and local-patch address calculations;
- balanced, compile-time GroupSum reduction.

No runtime-programmable weights, dynamic connection selection, floating-point
operators, RGB encoder, or additional learned layer is introduced. The bridge
also avoids constructing full im2col patch buffers: each LUT evaluation reads
only its six selected inputs.

The generated loop-based HLS description gives Vitis a representation in
which scheduling, latency, and resource sharing can be explored. It does not,
by itself, guarantee a particular reuse factor, initiation interval, routing
result, or performance. Those properties must be measured after C simulation,
HLS synthesis, and FPGA implementation.

### Engineering contribution and novelty

The contribution in this repository is the preservation-aware lowering of a
trained, irregular LUT6 neural network into hls4ml without approximating its
Boolean functions:

1. Learned input selection becomes immutable compile-time connection tables.
2. Learned 64-entry functions remain native LUT6 truth tables with the exact
   pin-order convention.
3. Boolean convolution weight sharing is represented explicitly instead of
   expanding the trained model into unrelated software operators.
4. Exact integer GroupSum classification is retained without floating-point
   inference.
5. A Python reference, custom hls4ml graph, and native circuit exporter provide
   independent representations for progressive equivalence checking.

This should be presented as the project's technical contribution rather than
as a claim that no related LUT/HLS method exists. The scientific question is
whether this exact lowering enables Vitis to find a more practical
latency-resource-routing trade-off than the fully combinational RTL.

## hls4ml workflow

Install hls4ml in the Python environment used for project generation. Vitis
HLS itself is not required merely to generate the C++ project.

### Step 1: export the trained model to `hls4ml_lut6_ir_v1`

```bash
env PYTHONPATH=src:. python -m \
  examples.phase_logic.KD_training.post_training.student_hls4ml_exporter \
  --checkpoint \
  examples/phase_logic/KD_training/runs/cifar10_tinyconv_q2_lbp2_dense2_kd_e500_b128_lr0.02_c16_tau40_seed42_a0.5_T2_dsfreeze_20261029T031327_251150Z/best_student.pt \
  --output \
  examples/phase_logic/KD_training/post_training/outputs/tinyconv_dense2_hls4ml/hls4ml_lut6_ir.json
```

### Step 2: generate the Vitis HLS C++ project

```bash
env PYTHONPATH=src:. python -m \
  examples.phase_logic.KD_training.post_training.hls4ml_bridge.convert_json \
  --json examples/phase_logic/KD_training/post_training/outputs/tinyconv_dense2_hls4ml/hls4ml_lut6_ir.json \
  --output-dir examples/phase_logic/KD_training/post_training/outputs/tinyconv_dense2_hls4ml/hls_project_v2 \
  --backend Vitis \
  --part xczu7ev-ffvc1156-2-e \
  --write
```

`--write` invokes hls4ml project generation only; it does not run C synthesis,
RTL synthesis, placement, or routing. The FPGA target may be changed later,
although regenerating the project with the desired `--part` is the safest
workflow.

### Optional reset-free banked input control (ZU7EV HLS target)

For the Q2-only model with 6,144 encoded input bits, add
`--bank-input-control` to the bridge command. This changes only the HLS core's
input protocol from `ap_vld` to `ap_none` and generates
`lut6_student_banked.v` beside the HLS project. The wrapper remains the
system-level top; `lut6_student` remains the HLS compute core. The wrapper
captures 64 banks of 96 bits using reset-free data registers and locally
registered load controls. Reset remains on the wrapper's control state and on
the core control. The 6,144-bit external port is still an internal/OOC FPGA
interface, not a package-pin interface.

```bash
env PYTHONPATH=src:. python -m \
  examples.phase_logic.KD_training.post_training.hls4ml_bridge.convert_json \
  --json examples/phase_logic/KD_training/post_training/outputs/tinyconv_q2_wide10_dense2_hls4ml/hls4ml_lut6_ir.json \
  --output-dir examples/phase_logic/KD_training/post_training/outputs/tinyconv_q2_wide10_dense2_hls4ml/hls_project_zu7ev_banked_control \
  --backend Vitis --part xczu7ev-ffvc1156-2-e \
  --bank-input-control --write
```

The wrapper accepts a frame when `ap_start && ap_ready && encoded_input_ap_vld`
is high at a rising edge. It starts the compute core two rising edges after
that accepted edge. `ap_ready` is low while a frame is in progress, so this
minimal wrapper accepts one frame at a time. The extra capture stages alter
transaction latency and may reduce throughput; check this in RTL simulation
before implementation. Learned LUT6 tables, connections, and spatial compute
instances are unchanged.

After generating the project, run HLS **from inside that project directory**:

```bash
cd examples/phase_logic/KD_training/post_training/outputs/tinyconv_q2_wide10_dense2_hls4ml/hls_project_zu7ev_banked_control
vitis-run --mode hls --tcl build_prj.tcl
```

`build_opt.tcl` enables `csynth_design`; it does not run Vivado placement or
routing. The HLS script uses a 5 ns **scheduling target**; it does not impose a
Vivado implementation clock constraint. After HLS, inspect the generated core
RTL and verify that it has no `encoded_input_ap_vld` input and no resettable
6,144-bit input holding register. Then include both the HLS-generated Verilog
files and `lut6_student_banked.v` in an out-of-context Vivado synthesis with
`lut6_student_banked` as top. Compare `report_high_fanout_nets` against the
original synthesized design **before** placement/routing. Finally run an RTL
testbench covering reset, accepted and ignored inputs, and consecutive frames;
confirm exact class counts and the two-edge wrapper launch delay before using
this version for implementation.

For a quick post-HLS fanout check from that project directory (no placement):

```bash
vivado -mode batch -source /dev/stdin <<'TCL'
read_verilog [glob lut6_student_prj/solution1/syn/verilog/*.v]
read_verilog lut6_student_banked.v
synth_design -mode out_of_context -top lut6_student_banked -part xczu7ev-ffvc1156-2-e
report_high_fanout_nets -max_nets 30 -file banked_fanout_synth.rpt
report_utilization -file banked_utilization_synth.rpt
TCL
```

### Localize the HLS core's dense-output clock enable

The HLS core's state-1 signal enables 8,800 Dense2 output register bits. After
`csynth_design`, use the repeatable RTL postprocessor to give each 64-bit
output bank its own protected enable register. It copies all HLS Verilog and
the existing 64-bank input wrapper to a separate directory; the HLS project
and learned weights remain unchanged. Each local enable follows the original
FSM next-state bit with the same reset and initialization value, so it controls
the same clock edges without adding a data-path stage.

```bash
env PYTHONPATH=src:. python -m \
  examples.phase_logic.KD_training.post_training.hls4ml_bridge.localize_fsm_ce \
  --project-dir examples/phase_logic/KD_training/post_training/outputs/tinyconv_q2_wide10_dense2_hls4ml/hls_project_zu7ev_banked_control \
  --output-dir examples/phase_logic/KD_training/post_training/outputs/tinyconv_q2_wide10_dense2_hls4ml/hls_project_zu7ev_banked_control/rtl_local_ce_v1
```

Add **only** the `.v` files from `rtl_local_ce_v1` to a Vivado project and set
`lut6_student_banked` as top. Do not add the original HLS Verilog at the same
time: it defines the same modules. The postprocessor checks the exact source
pattern and fails if HLS changes its output-register structure. The separate
`local_ce_report.json` records the source hashes and bank counts. Verify the
local enable fanout in a synthesized netlist before running implementation;
the RTL bank count alone does not establish a physical routing improvement.

Copy the complete `hls_project_v2` directory to the Windows Vitis machine. The
main C++ file depends on its generated headers, connection/truth-table
constants, configuration files, and TCL scripts.

### Optional C-simulation equivalence check

When the local hls4ml C-simulation toolchain is available, compare generated C
against the Python reference and previously saved circuit vectors:

```bash
env PYTHONPATH=src:. python -m \
  examples.phase_logic.KD_training.post_training.hls4ml_bridge.convert_json \
  --json examples/phase_logic/KD_training/post_training/outputs/tinyconv_dense2_hls4ml/hls4ml_lut6_ir.json \
  --output-dir examples/phase_logic/KD_training/post_training/outputs/tinyconv_dense2_hls4ml/hls_project_v2 \
  --backend Vitis \
  --part xcvu19p-fsva3824-2-e \
  --compare-npz examples/phase_logic/KD_training/post_training/outputs/tinyconv_dense2_dsfreeze_seed42_epoch360/circuit/verification_vectors.npz \
  --samples 4
```

This compiles and runs C simulation; it still does not start HLS synthesis.

## Verification status

The current evidence must be interpreted conservatively:

- The native circuit exporter checks exact integer class counts before and
  after simplification and after JSON reload.
- The hls4ml bridge's Python reference matched four existing checkpoint-derived
  verification vectors exactly.
- The hls4ml ModelGraph and project-generation path has been implemented and
  corrected for backend registration and template formatting issues.
- Successful generated-C compilation and C-simulation equivalence still need
  to be recorded for the target hls4ml/Vitis environment.
- Vitis HLS synthesis, RTL co-simulation, timing, resource use, placement, and
  routing have not yet been established for this bridge.

Therefore, the repository does not yet claim that the HLS result is bit-exact,
synthesizable, timing-clean, or routable. These are explicit next validation
milestones.

## Current scope and limitations

The bridge currently accepts exactly:

```text
2 x LUT6Conv2D -> 2 x LUT6Dense -> 1 x BooleanGroupSum
```

The widths and channel counts are read from JSON and may change, provided every
node remains LUT6 and all connection ranges and shapes remain valid. The
GroupSum size must fit the unsigned 10-bit output. A model with three
convolutions, three dense layers, LUT2 gates, branches, skip connections, or a
different operation order requires extending the validator/reference graph.
The underlying LUT6 HLS primitives can be reused, but the current frontend does
not accept those graph structures automatically.
