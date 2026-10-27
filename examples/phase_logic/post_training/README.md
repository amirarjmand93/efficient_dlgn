# Checkpoint loading and architecture visualization

Rebuild a CompactLogic CIFAR-10 model using its saved configuration, restore
its learned gate candidates and connections, and switch to discrete evaluation.
The loader does not train or download datasets. Circuit export is a separate
command described below.

```bash
python -m examples.phase_logic.post_training.load_checkpoint \
  --checkpoint "path/to/best_model.pt" \
  --output-dir examples/phase_logic/post_training/outputs/my_model
```

Outputs are `architecture.png`, a scalable `architecture.svg`, and
`architecture.json` containing configuration, counts, shapes, and graph edges.
The diagram traces the actual Python forward wiring, treating candidate layers
as leaves, so concatenation skips and MRLB branches are visible without
evaluating the model. It is a layer-level graph, not a synthesized gate netlist.
Shapes omit batch size. Without `--output-dir`, files go into `post_training/`
beside the checkpoint. Rerunning with the same output directory replaces these
inspection artifacts, not the checkpoint.

Defaults are `--device cpu --backend torch`; the installed CompactLogic CUDA
extension is not needed. CUDA-trained checkpoints are supported: only known
CUDA inverse-connectivity acceleration buffers are filtered for the reference
backend. Learned input indices, offsets, opcodes, logits, and frozen masks are
strictly restored. `--device cuda --backend cuda` uses the upstream extension.
`--smoke-test` optionally runs one synthetic image, not an accuracy evaluation.

Python usage:

```python
from examples.phase_logic.post_training.load_checkpoint import load_checkpoint

model, report = load_checkpoint("path/to/best_model.pt")
# model.eval() is already set; pass float RGB images [N,3,32,32] in [0,1].
# Run inference inside torch.inference_mode() when gradients are not needed.
```

Training checkpoints use embedded `config`. A raw `final_model.pt` state dict
uses sibling `config.json`; `--config path/to/config.json` explicitly overrides
either source. A sibling `model_report.json` restores GroupSum tau when saved,
since tau is not a state-dict tensor. Without it, the current builder's default
is used with a warning. All files must come from the same run. Optimizer,
controller, and saved RNG states are not restored; this is not a resume tool.

Current supported architectures are the registered CompactLogic CIFAR-10
variants, including shared heads, wide75, six-conv, and MRLB. Incompatible older
builder revisions (for example a historical three-dense version under a reused
model name) are rejected explicitly, not silently approximated. Preserve the
source revision matching each run. Model restoration uses
`torch.load(..., weights_only=True)` and never enables unsafe pickle fallback.

## Export existing Torchlogix Circuit IR, C, and Verilog

From the repository root:

```bash
env PYTHONPATH=src python -m examples.phase_logic.post_training.export_circuit \
  --checkpoint "path/to/best_model.pt" \
  --output-dir examples/phase_logic/post_training/outputs/my_model/circuit
```

`env PYTHONPATH=src` works in bash and tcsh and selects this checkout's
`src/torchlogix` instead of an older installed package. Alternatively install
this checkout in editable mode. The exporter checks that the active Verilog
backend preserves constant contributions folded into GroupSum; old emitters
with the missing-offset bug are rejected rather than silently producing wrong RTL.

Flow:

```text
checkpoint -> existing loader -> selected candidate opcodes and pairs
  -> existing Circuit/Gate/SumReduction IR -> verify counts -> simplify
  -> verify counts -> JSON / C / Verilog
```

The adapter traces only module-level wiring, not CUDA kernels. It unrolls each
selected conv kernel at every output position, applies the selected spatial
offsets and zero padding, preserves concatenation/flatten wiring, and respects
frozen dense choices. The original model and checkpoint are not modified.

Outputs:

- `circuit.json`: simplified gate-level IR, reloadable with `Circuit.from_json_file`.
- `circuit.c`: existing C backend, `void circuit(...)`, Boolean input array.
- `circuit.v`: existing combinational Verilog backend, top module `circuit`.
- `export_report.json`: interface contract, encoder parameters, score transform,
  gate counts, warnings, and verification status.
- `verification_vectors.npz`: encoded inputs and expected integer class counts.

**The interface is encoded Boolean CHW input, not RGB.** For Q2+LBP2 this is
`12 x 32 x 32 = 12,288` input bits, in the encoder's actual channel order.
The encoder remains separate; saved thresholds, feature parameters, LBP rules,
and channel order are recorded in the report. The tool does not generate encoder
RTL or C. Pass one image flattened in CHW order to the C function; Verilog input
bit `inp[i]` is the same flattened feature `i` (bit 0 is the first feature).

**Outputs are integer class counts**, with 10 outputs for CIFAR-10, not scaled
logits. Recover original logits with `(counts + source_beta) / source_tau`, using
the report. A finite positive shared tau and finite shared beta preserve argmax
and ties; other score transformations are rejected. In Verilog class `j` occupies
the `j`th score slice starting at bit `j * score_bits`; the generated declaration
and comments state the integer width. Internal reduction offsets created during
simplification are retained in both C and Verilog.

By default four synthetic RGB examples (black, white, and two random images)
verify exact counts against the loaded model before and after simplification
and after JSON reload. This is a functional smoke check, not a dataset accuracy
evaluation or formal equivalence proof. Use `--verify-samples N` for more examples.

Optional `--compile-c --c-opt-level 0` compiles with GCC and verifies generated C
counts too. Compiling a large fully unrolled model may take substantial time
and memory; it is not enabled by default. Existing CPU packing can be accessed
from the resulting `Circuit` API; it packs different samples, not neighboring
features of one image. This is not yet an optimized MCU runtime.

For FPGA synthesis, use the existing workflow after installing Vivado:

```bash
vivado -mode batch -source examples/synthesis/synthesize.tcl \
  -tclargs path/to/circuit.v YOUR_FPGA_PART path/to/synthesis_reports
```

This produces synthesis estimates, not board deployment or post-route timing.
The exported core has no clock, pipeline, memory interface, input encoder, or
physical pin mapping. Verilator/Vivado availability and FPGA fit are separate
from successful Python/IR/C checks. Counts after IR simplification are not FPGA
LUT counts. Keep `circuit.v` named as shown: the script derives top from filename.
