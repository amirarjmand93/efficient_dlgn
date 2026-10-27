# ELAR GroupSum hierarchical RTL experiment

This is a **post-export experiment**, designed for the existing `circuit.json`, `circuit.v`, and `verification_vectors.npz` produced by `examples.phase_logic.post_training.export_circuit`.

It uses the [Torchlogix `Circuit.get_verilog_code()` implementation at commit `82e5b48`](https://github.com/ligerlac/torchlogix/blob/82e5b48da697252dcce5f53467040931cb366c19/src/torchlogix/circuit.py) as the reference.

## 1. Generate an alternative RTL file

Run from the repository root and pass the original `circuit.v`:

```bash
python examples/phase_logic/post_training/modified_verilog_pr/generate_hierarchical_groupsum.py \
    "/path/to/original-export/circuit.v"
```

The tool automatically reads `circuit.json` beside the input and creates a
uniquely named directory under `modified_verilog_pr/generated/`. The output file
and top module are both still named `circuit`. The original file cannot be used
as the output path.

To choose a readable destination name:

```bash
python examples/phase_logic/post_training/modified_verilog_pr/generate_hierarchical_groupsum.py \
    "/path/to/original-export/circuit.v" \
    --output-dir examples/phase_logic/post_training/modified_verilog_pr/generated/my_run
```

Use `--ir /other/path/circuit.json` only when the matching JSON is not beside
the Verilog. Existing generated output is protected unless `--force` is passed.
Each run also creates `transformation_report.json` with source/output hashes and
the input, constant, and chunk count for every class.

### Dense-1 -> Dense-2 locality option

If GroupSum hierarchy alone still routes poorly, enable connectivity-local
tiling for the final two Boolean stages:

```bash
python examples/phase_logic/post_training/modified_verilog_pr/generate_hierarchical_groupsum.py \
    "/path/to/original-export/circuit.v" \
    --localize-dense \
    --dense-tile-size 512 \
    --output-dir examples/phase_logic/post_training/modified_verilog_pr/generated/my_locality_run
```

The tool finds connected components of the Dense-1 -> Dense-2 bipartite graph,
packs whole components into roughly 512-gate modules, and applies
`keep_hierarchy` to those modules. Consequently, no Dense-1 -> Dense-2 edge
crosses a tile boundary. This changes hierarchy only: every Boolean equation
and connection remains present exactly once, and no buffers, registers, clocks,
or pipeline stages are introduced. The audit report includes fanout/locality
measurements and per-tile statistics.

This is a placement hint, not a routing guarantee. Use a separate Vivado run
with the same device, constraints, and implementation strategy when comparing
it with the baseline.

The alternative RTL uses 128-bit local balanced population counters and a balanced reduction tree per class. No registers are added. The circuit gates, encoder exclusion, class order and score width remain unchanged. Importantly, integer `SumReduction.beta` offsets folded by simplification are explicitly included.

Optional `--keep-hierarchy` adds hierarchy attributes to popcount instances. Test this as a separate configuration; it is **not** enabled by default and may change LUT usage.

## 2. Generate exact-count verification testbench

```bash
python make_groupsum_verification_tb.py \
    --ir "/path/to/old-export/circuit.json" \
    --vectors "/path/to/old-export/verification_vectors.npz" \
    --output "/path/to/new-hierarchical/tb_counts.v"
```

If Icarus Verilog (`iverilog`, `vvp`) is installed:

```bash
iverilog -g2012 -s tb_counts -o sim.vvp \
    /path/to/new-hierarchical/circuit.v \
    /path/to/new-hierarchical/tb_counts.v
vvp sim.vvp
```

This testbench checks **each raw integer class score**, not just argmax, on the previously exported verification vectors. Four provided vectors are a smoke test rather than an exhaustive equivalence proof; use additional samples and/or formal sequential/combinational equivalence checking if available.

The transformer itself also performs strict pre-write checks: the behavioral
GroupSum ranges and folded offsets must match `circuit.json`, and every `raw[]`
assignment must point to the same IR input in the same order. It refuses unknown
Verilog output formats instead of making a best-effort rewrite.

Run its dependency-free tests with:

```bash
python examples/phase_logic/post_training/modified_verilog_pr/test_generate_hierarchical_groupsum.py
```

## 3. Controlled Vivado experiment

Use the **same FPGA (`xcvu19p-fsva3824-2-e`), project settings, flattening policy and constraints** as the original. Run a fresh, nonincremental synthesis (avoid reusing an unrelated reference checkpoint) and attempt placement/routing. Compare LUT primitives, physical CLB LUT usage, congestion, routed status, and timing if constraints are defined. A different GroupSum structure **does not guarantee congestion relief**: Dense1-to-Dense2 interconnect may still dominate.

Do not replace the active reference `circuit.v` before successful RTL simulation. The new script changes only the Verilog output, not the `circuit.json` or `circuit.c`.
