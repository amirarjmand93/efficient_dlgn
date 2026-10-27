"""Export selected CompactLogic gates to the existing Torchlogix Circuit IR.

Input is already-encoded Boolean CHW data, not raw RGB. Outputs are raw integer
class counts. For the original logits: (counts + source_beta) / source_tau.
Positive shared tau and shared beta preserve class argmax, including ties.
"""

import argparse
import json
import math
from pathlib import Path
import re

import numpy as np
import torch

from torchlogix.circuit import Circuit, Gate, GateOp, LUT_ID_TO_GATE, SumReduction
from torchlogix.layers import GroupSum

from ..compact import CandidateLogicLayer, UniformThermometerEncoder
from ..model import HybridFeatureEncoder
from .load_checkpoint import ArchitectureTracer, load_checkpoint


def _selected_candidates(layer):
    """Read the actual opcode/pair, never treat a candidate slot as an opcode."""
    logits = layer.weights.detach().cpu()
    if logits.ndim != 2 or not bool(torch.isfinite(logits).all()):
        raise ValueError("Candidate logits must be a finite [outputs,candidates] matrix.")
    slots = getattr(layer.impl, "w", logits.argmax(-1)).detach().cpu().long()
    if slots.shape != (layer.out_dim,) or bool(((slots < 0) | (slots >= logits.shape[1])).any()):
        raise ValueError("Invalid cached candidate selection.")
    rows = torch.arange(layer.out_dim)
    gates = layer.impl.gate_sequence.detach().cpu()
    if gates.shape != logits.shape:
        raise ValueError("Candidate opcode/logit shape mismatch.")
    codes = gates[rows, slots].long().clone()
    if layer.is_convolution:
        pairs = tuple(getattr(layer.impl, "offsets_" + axis).detach().cpu()[:, rows, slots].long()
                      for axis in ("ch", "h", "w"))
    else:
        pairs = layer.impl.indices.detach().cpu()[:, rows, slots].long().clone()
        if hasattr(layer, "frozen_rows"):
            mask = layer.frozen_rows.detach().cpu()
            codes[mask] = layer.frozen_codes.detach().cpu()[mask].long()
            pairs[:, mask] = layer.frozen_pairs.detach().cpu()[:, mask].long()
        if pairs.shape != (2, layer.out_dim) or bool(((pairs < 0) | (pairs >= layer.impl.in_dim)).any()):
            raise ValueError("Selected dense connections escape the input representation.")
    if codes.shape != (layer.out_dim,) or bool(((codes < 0) | (codes > 15)).any()):
        raise ValueError("Selected Boolean operation IDs must be in 0..15.")
    return codes.numpy(), pairs


def compact_to_circuit(model, *, input_shape=None):
    """Build a Circuit directly from discrete candidates and FX wiring.

    Reuses existing Circuit/Gate/SumReduction types. Encoders are bypassed, conv
    kernels are expanded in CHW order, zero padding is an explicit FALSE wire,
    and cat/flatten are wiring only. Does not mutate model or call its kernels.
    """
    if model.training:
        raise ValueError("Export requires model.eval(); relaxed mixtures cannot be exported as Boolean gates.")
    modules = dict(model.named_modules())
    encoders = [module for module in modules.values()
                if isinstance(module, (HybridFeatureEncoder, UniformThermometerEncoder))]
    if len(encoders) != 1:
        raise ValueError("Expected exactly one input encoder.")
    if input_shape is None:
        input_shape = (encoders[0].output_channels, 32, 32)
    input_shape = tuple(input_shape)
    if len(input_shape) != 3 or input_shape[0] != encoders[0].output_channels or min(input_shape) <= 0:
        raise ValueError("input_shape must be encoded (channels,height,width), matching the encoder.")
    circuit = Circuit(n_inputs=math.prod(input_shape), input_shape=list(input_shape))
    graph = ArchitectureTracer().trace(model)
    wires = {}
    next_id = circuit.n_inputs
    padding_id = None
    terminal_sum = None

    def false_wire():
        nonlocal next_id, padding_id
        if padding_id is None:
            padding_id = next_id
            circuit.gates.append(Gate(next_id, GateOp.CONST_FALSE))
            next_id += 1
        return padding_id

    def append_gates(codes, a, b, shape, layer_index):
        nonlocal next_id
        count = math.prod(shape)
        if len(codes) != count or a.size != count or b.size != count:
            raise ValueError("Candidate output/wiring alignment mismatch.")
        outputs = np.arange(next_id, next_id + count, dtype=np.int64).reshape(shape)
        for code, a_id, b_id in zip(codes, a.reshape(-1), b.reshape(-1)):
            operation, uses_a, uses_b = LUT_ID_TO_GATE[int(code)]
            in0 = int(a_id) if uses_a else int(b_id) if uses_b else -1
            in1 = int(b_id) if uses_a and uses_b else -1
            circuit.gates.append(Gate(next_id, operation, in0, in1, layer_index))
            next_id += 1
        return outputs

    for layer_index, node in enumerate(graph.nodes):
        if terminal_sum is not None and node.op != "output":
            raise ValueError("Only a final GroupSum followed directly by output is supported.")
        if node.op == "placeholder":
            if wires:
                raise ValueError("Only one model input is supported.")
            wires[node] = np.arange(circuit.n_inputs, dtype=np.int64).reshape(input_shape)
        elif node.op == "call_module":
            module = modules[node.target]
            source = wires[node.args[0]]
            if isinstance(module, (HybridFeatureEncoder, UniformThermometerEncoder)):
                if node.args[0].op != "placeholder":
                    raise ValueError("Encoder must consume the raw model input directly.")
                wires[node] = source  # External preprocessing already produced these bits.
            elif isinstance(module, torch.nn.Flatten):
                if module.start_dim != 1 or module.end_dim != -1:
                    raise ValueError("Only per-image flatten(start_dim=1) is supported.")
                wires[node] = source.reshape(-1)
            elif isinstance(module, CandidateLogicLayer):
                codes, selected = _selected_candidates(module)
                if module.is_convolution:
                    if tuple(source.shape) != tuple(module.in_shape):
                        raise ValueError(f"{node.target}: convolution input shape mismatch.")
                    channels, height, width = source.shape
                    ch, dh, dw = (value.numpy() for value in selected)
                    radius = module.impl.padding
                    if (ch.shape != (2, module.out_dim) or bool(((ch < 0) | (ch >= channels)).any())
                            or bool((np.abs(dh) > radius).any()) or bool((np.abs(dw) > radius).any())):
                        raise ValueError(f"{node.target}: selected spatial offsets are out of bounds.")
                    hs = np.arange(0, height, module.stride)
                    ws = np.arange(0, width, module.stride)
                    operands = []
                    for side in range(2):
                        h = hs[None, :, None] + dh[side, :, None, None]
                        w = ws[None, None, :] + dw[side, :, None, None]
                        valid = (h >= 0) & (h < height) & (w >= 0) & (w < width)
                        values = source[ch[side, :, None, None], np.clip(h, 0, height - 1),
                                        np.clip(w, 0, width - 1)]
                        if not valid.all():
                            values = np.where(valid, values, false_wire())
                        operands.append(values)
                    shape = (module.out_dim, len(hs), len(ws))
                    wires[node] = append_gates(np.repeat(codes, len(hs) * len(ws)),
                                               *operands, shape, layer_index)
                else:
                    if source.ndim != 1 or source.size != module.impl.in_dim:
                        raise ValueError(f"{node.target}: dense input shape mismatch.")
                    pairs = selected.numpy()
                    wires[node] = append_gates(codes, source[pairs[0]], source[pairs[1]],
                                               (module.out_dim,), layer_index)
            elif isinstance(module, GroupSum):
                tau, beta = float(module.tau), float(module.beta)
                if not math.isfinite(tau) or tau <= 0 or not math.isfinite(beta):
                    raise ValueError("Raw-count export requires finite, positive shared tau and finite shared beta.")
                if source.ndim != 1 or source.size % module.k:
                    raise ValueError("GroupSum requires a class-divisible flat feature vector.")
                for group in source.reshape(module.k, -1):
                    circuit.sum_nodes.append(SumReduction(next_id, group.tolist(), tau=1.0, beta=0.0))
                    circuit.outputs.append(next_id)
                    next_id += 1
                circuit.output_shape = [module.k]
                terminal_sum = node
            else:
                raise ValueError(f"Unsupported export module: {node.target} ({type(module).__name__}).")
        elif node.op == "call_function" and node.target is torch.cat:
            dimension = node.kwargs.get("dim", node.args[1] if len(node.args) > 1 else 0)
            if dimension != 1:
                raise ValueError("Only channel/feature concatenation (dim=1) is supported.")
            wires[node] = np.concatenate([wires[source] for source in node.args[0]], axis=0)
        elif node.op == "output":
            if terminal_sum is None or node.args[0] is not terminal_sum:
                raise ValueError("Model output must be a single final GroupSum.")
        else:
            raise ValueError(f"Unsupported export graph operation: {node.op} {node.target}.")
    if terminal_sum is None:
        raise ValueError("No final GroupSum found.")
    return circuit


def verification_data(model, *, samples=4, seed=0):
    """Generate encoded inputs and exact reference counts from full RGB inference."""
    if samples < 1:
        raise ValueError("At least one verification sample is required.")
    generator = torch.Generator().manual_seed(seed)
    rgb = torch.rand(samples, 3, 32, 32, generator=generator)
    if samples >= 2:
        rgb[0].zero_()
        rgb[1].fill_(1)
    device = next(model.parameters()).device
    rgb = rgb.to(device)
    encoder = next(module for module in model.modules()
                   if isinstance(module, (HybridFeatureEncoder, UniformThermometerEncoder)))
    groups = [module for module in model.modules() if isinstance(module, GroupSum)]
    if len(groups) != 1:
        raise ValueError("Expected one final GroupSum.")
    group_sum = groups[0]
    captured = []

    def capture(module, args):
        features = args[0]
        if not bool(((features == 0) | (features == 1)).all()):
            raise ValueError("Reference model did not produce hard Boolean features.")
        captured.append(features.reshape(samples, module.k, -1).sum(-1).cpu().long())

    hook = group_sum.register_forward_pre_hook(capture)
    try:
        with torch.inference_mode():
            encoded = encoder(rgb).cpu().bool().numpy()
            logits = model(rgb).cpu()
    finally:
        hook.remove()
    if len(captured) != 1:
        raise ValueError("Final GroupSum was not evaluated exactly once.")
    counts = captured[0]
    torch.testing.assert_close((counts.float() + group_sum.beta) / group_sum.tau, logits)
    return encoded, counts.numpy(), {"tau": float(group_sum.tau), "beta": float(group_sum.beta)}


def verify_circuit(circuit, encoded, expected, *, compiled=False):
    """Check exact counts, not merely classification agreement."""
    if tuple(encoded.shape[1:]) != tuple(circuit.input_shape):
        raise ValueError("Verification encoder shape differs from circuit inputs.")
    actual = circuit(encoded, use_compiled=compiled)
    actual = actual.numpy() if isinstance(actual, torch.Tensor) else actual
    if not np.array_equal(actual, expected):
        error = np.max(np.abs(actual.astype(np.int64) - expected.astype(np.int64)))
        raise ValueError(f"Circuit verification failed: max class-count difference {error}.")
    return {"samples": len(encoded), "exact_class_counts": True, "same_argmax": True}


def export_checkpoint(checkpoint, output_dir, *, config_path=None, verify_samples=4,
                      compile_c=False, c_opt_level=0, simplify_passes=100):
    """Load on CPU, verify -> simplify -> verify -> emit existing IR/C/Verilog."""
    if verify_samples < 1 or simplify_passes < 1 or c_opt_level not in (0, 1, 2, 3):
        raise ValueError("Verification samples and simplify passes must be positive; C optimization is 0..3.")
    output_dir = Path(output_dir).expanduser().resolve()
    model, model_report = load_checkpoint(checkpoint, config_path=config_path, device="cpu", backend="torch")
    print("Loaded checkpoint; converting selected gates to encoded-input Circuit IR...", flush=True)
    circuit = compact_to_circuit(model)
    encoded, expected, score_transform = verification_data(model, samples=verify_samples)
    before = len(circuit.gates)
    print(f"Built {before:,} IR gates; verifying unsimplified class counts...", flush=True)
    checks = {"before_simplification": verify_circuit(circuit, encoded, expected)}
    print("Simplifying with torchlogix.Circuit.simplify()...", flush=True)
    circuit.simplify(n_max=simplify_passes)
    checks["after_simplification"] = verify_circuit(circuit, encoded, expected)
    print(f"Verified simplified IR: {len(circuit.gates):,} gates. Generating JSON/C/Verilog...", flush=True)
    # Generators are the existing source backends. Always name RTL circuit.v:
    # its module is 'circuit', and the Vivado script derives top from filename.
    c_code = circuit.get_c_code()
    verilog = circuit.get_verilog_code()
    # A non-editable installed torchlogix may predate the local emitter fix.
    # Do not silently write RTL that drops TRUE inputs folded into sum offsets.
    sums = {sr.node_id: sr for sr in circuit.sum_nodes}
    for index, output_id in enumerate(circuit.outputs):
        sr = sums[output_id]
        if sr.input_ids and sr.beta and not re.search(
            rf"\bs_{index}\s*=\s*{int(sr.beta)}\s*;", verilog
        ):
            raise ValueError("Active torchlogix Verilog emitter drops folded constant counts. "
                             "Use the updated source: env PYTHONPATH=src python -m "
                             "examples.phase_logic.post_training.export_circuit ... "
                             "or install this checkout in editable mode.")
    output_dir.mkdir(parents=True, exist_ok=True)
    circuit.write_json(str(output_dir / "circuit.json"))
    (output_dir / "circuit.c").write_text(c_code)
    (output_dir / "circuit.v").write_text(verilog)
    reloaded = Circuit.from_json_file(str(output_dir / "circuit.json"))
    checks["json_roundtrip"] = verify_circuit(reloaded, encoded, expected)
    if compile_c:
        print("Compiling C with GCC and verifying class counts...", flush=True)
        circuit.compile(opt_level=c_opt_level)
        checks["compiled_c"] = verify_circuit(circuit, encoded, expected, compiled=True)
    encoder = model[0]
    encoder_state = {name: value.detach().cpu().tolist() for name, value in encoder.state_dict().items()}
    report = {
        "checkpoint": str(Path(checkpoint).expanduser().resolve()),
        "model_variant": model_report["model_variant"], "step": model_report["step"],
        "input": {"kind": "encoded Boolean bits, NOT raw RGB", "shape": circuit.input_shape,
                  "layout": "CHW, row-major; flattened index=(channel*height+y)*width+x",
                  "encoder_configuration": {key: model_report["configuration"].get(key) for key in
                                            ("input_encoding", "hybrid_features", "num_input_bits",
                                             "num_quantile_bits", "num_sine_bits", "sine_sigma", "sine_seed")},
                  "encoder_state": encoder_state,
                  "channel_order": encoder.channel_order,
                  "lbp": {"radius": 1, "padding": "replicate", "comparisons":
                          ["right >= left", "down >= up"]} if getattr(encoder, "num_lbp_bits", 0) else None},
        "output": {"kind": "raw integer class counts", "shape": circuit.output_shape,
                   "original_logits": "(counts + beta) / tau", **score_transform},
        "gates_before_simplification": before, "gates_after_simplification": len(circuit.gates),
        "gate_count_note": "Spatially expanded; includes shared padding constant before simplification. Encoder and GroupSum arithmetic excluded.",
        "verification": checks, "warnings": model_report["warnings"],
        "rtl_note": "Combinational encoded-input core only; no clock, pipeline, board I/O, encoder, or place-and-route.",
    }
    (output_dir / "export_report.json").write_text(json.dumps(report, indent=2) + "\n")
    # Integer text vectors are directly usable when adapting the provided testbench.
    np.savez_compressed(output_dir / "verification_vectors.npz", encoded=encoded, class_counts=expected)
    return circuit, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--verify-samples", type=int, default=4,
                        help="Synthetic RGB examples; checks exact counts before/after simplification (default 4).")
    parser.add_argument("--simplify-passes", type=int, default=100)
    parser.add_argument("--compile-c", action="store_true", help="Also compile with GCC and verify the generated C.")
    parser.add_argument("--c-opt-level", type=int, choices=(0, 1, 2, 3), default=0,
                        help="GCC optimization for optional compiled check; large circuits can take substantial time.")
    args = parser.parse_args()
    try:
        circuit, report = export_checkpoint(
            args.checkpoint, args.output_dir, config_path=args.config,
            verify_samples=args.verify_samples, compile_c=args.compile_c,
            c_opt_level=args.c_opt_level, simplify_passes=args.simplify_passes,
        )
    except (ValueError, RuntimeError, OSError, ImportError, AssertionError) as error:
        parser.exit(1, f"Error: {error}\n")
    print(f"Exported {report['gates_before_simplification']:,} -> {len(circuit.gates):,} gates.")
    print(f"Saved circuit.json, circuit.c, circuit.v, export_report.json and verification_vectors.npz in {args.output_dir.resolve()}")
    print("Inputs are encoded bits. Outputs are integer class counts; encoder and FPGA board integration are separate.")


if __name__ == "__main__":
    main()
