# LUT6 graph to sequential SystemVerilog

The current teacher flow uses `graph_compiler.py` to turn **our project's
packed LUT6 graph JSON** into a sequential FPGA project.
Run the commands below from the **project root**. The separate
`hls4ml_lut6_ir_v1` layer compiler is documented at the end.

## Current teacher flow

```text
best_model.pt
  → teacher_lut6_exporter.py → lut6_ir.json
  → graph_compiler.py → graph_sv/
      ├── rtl/                  sequential engine and nine ROM files
      ├── sim/ and vectors/     self-checking XSim testbench and inputs
      ├── vivado/implement.tcl  synthesis, placement, routing
      ├── manifest.json         graph and ROM dimensions
      └── report_inference/     latency report after XSim passes
```

The checkpoint exporter is described in the [parent README](../README.md).
If you already have `lut6_ir.json`, compile it directly. For example:

```bash
env PYTHONPATH=examples/phase_logic/New_post_training/lut6_json_sv_bridge \
  python -m lut6_json_bridge.graph_compiler \
  --ir examples/phase_logic/New_post_training/outputs/best_cifar10_no_cai/lut6_ir.json \
  --out examples/phase_logic/New_post_training/outputs/best_cifar10_no_cai/graph_sv_current \
  --random-tests 1
```

Replace `--ir` and `--out` for another run. `--random-tests` adds random
Boolean vectors to the three built-in vectors. `--clock-ns` sets both the XSim
testbench period and the Vivado timing constraint (default: 10 ns).
`--debug-scores` exposes the full class-score bus and makes XSim check exact
scores as well as the predicted class.

### What the graph compiler emits

The top-level interface streams **pre-encoded Boolean input bits** in the
JSON's order, one bit per clock. After `start`, the engine evaluates graph
nodes sequentially and accumulates exact integer GroupSum scores. It outputs
registered `predicted_class` and a `pred_valid` pulse. Class width is derived
from the JSON; equal scores select the lowest class index. Full scores remain
internal unless `--debug-scores` is used.

ROMs hold node arity, connections, truth tables, and GroupSum references.
Two-input records contain their four truth bits directly. Other gate records
index a shared truth-table ROM. Each GroupSum record combines its source
address and inversion bit. An indexed bit RAM holds input and intermediate
values. The generated `rtl/` directory has **nine `.mem` files**; keep them
with the SystemVerilog source.

At compile time, the bridge checks ROM serialization and compares exact class
scores from its program evaluator with the original JSON evaluator on sampled
vectors. These checks do not prove equivalence for every possible input.

### Verify functionality and inspect latency

In Bash, with Vivado simulation tools available:

```bash
cd examples/phase_logic/New_post_training/outputs/best_cifar10_no_cai/graph_sv_current
source /home/j3k9j/AMD/2026.1/Vivado/settings64.sh
./run_sim.sh
```

XSim checks `predicted_class` and `pred_valid` for every generated vector. With
`--debug-scores`, it checks every class score too. A passing run writes:

- `report_inference/summary.md`: case count, clock period, graph size, and
  average/range of inference latency.
- `report_inference/latency.csv`: per-case cycles and simulated nanoseconds
  from accepted `start` and from the first accepted input bit to `pred_valid`.

The reported milliseconds use the **testbench clock**, not post-route FPGA
timing. Input encoding and external transfers are excluded. Older generated
projects must be regenerated to gain this report.

### Synthesize and route

From the same generated project directory:

```bash
vivado -mode batch -source vivado/implement.tcl \
  -tclargs xc7z035ffg676-2 z7035
```

The script writes `reports_z7035/`, including utilization, route status,
routed timing and power reports, plus `routed_design.dcp`. The manifest's BRAM
estimate is only a ROM-layout estimate; use Vivado reports for FPGA resources
and critical delay. This project has no board-level input encoder, I/O timing
constraints, PS interface, or bitstream deployment flow.

## Legacy layer-oriented compiler

`lut6_json_bridge.compiler` remains available for the separate
`hls4ml_lut6_ir_v1` format. It supports a **linear chain** of `LUT6Conv2D` and
`LUT6Dense` stages followed by `BooleanGroupSum`; each compute stage reads its
immediate predecessor. It does not accept the current teacher's native graph
JSON or its arbitrary skip/branch connections.

From the bridge directory, its included reference example is:

```bash
cd examples/phase_logic/New_post_training/lut6_json_sv_bridge
python -m lut6_json_bridge.compiler \
  --ir examples/reference_original_ir.json \
  --out generated_original --encoding full
```

`--encoding full` stores each LUT's INIT and six addresses in one instruction.
`--encoding split` uses a separate shared truth-table ROM; it can reduce raw
ROM bits but changes cycle count and may not reduce FPGA BRAM usage. This path
also generates its own simulation and Vivado scripts, including Windows
PowerShell runners. Its supported inputs are pre-encoded Boolean tensors;
raw-image encoding and non-Boolean operations are outside this compiler.
