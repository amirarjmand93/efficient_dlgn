"""Export the two-dense LUT6 student as a native combinational circuit.

The generated core consumes externally encoded Q2+LBP2 Boolean bits in CHW
order and returns ten unsigned raw class counts. It has no clock or pipeline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torchlogix.circuit import (
    LUT6Gate,
    LUT6Signal,
    LUT6SumReduction,
    NativeLUT6Circuit,
)

from .load_checkpoint import load_student_checkpoint


INPUT_SHAPE = (12, 32, 32)
SCORE_BITS = 16


def _as_numpy(value, dtype=None):
    result = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
    return result.astype(dtype, copy=False) if dtype is not None else result


def _init_values(tables) -> list[int]:
    bits = _as_numpy(tables, np.uint8)
    powers = np.left_shift(np.uint64(1), np.arange(64, dtype=np.uint64))
    return [int(value) for value in (bits.astype(np.uint64) * powers).sum(1, dtype=np.uint64)]


def _dense_eval(inputs: np.ndarray, connections, tables) -> np.ndarray:
    connections = _as_numpy(connections, np.int64)
    truth = _as_numpy(tables, np.bool_)
    selected = inputs[:, connections]
    address = (selected.astype(np.uint8) << np.arange(6, dtype=np.uint8)).sum(-1)
    nodes = np.arange(connections.shape[0])[None, :]
    return truth[nodes, address].astype(np.uint8)


def _conv_eval(inputs: np.ndarray, layer: dict) -> np.ndarray:
    connections = _as_numpy(layer["connections"], np.int64)
    truth = _as_numpy(layer["truth_tables"], np.bool_)
    batch, in_channels, height, width = inputs.shape
    out_channels = connections.shape[0]
    out_height, out_width = (height + 1) // 2, (width + 1) // 2
    output = np.zeros((batch, out_channels, out_height, out_width), dtype=np.uint8)
    channels = connections // 9
    remainder = connections % 9
    kernel_y, kernel_x = remainder // 3, remainder % 3
    for out_y in range(out_height):
        for out_x in range(out_width):
            values = np.zeros((batch, out_channels, 6), dtype=np.uint8)
            for pin in range(6):
                ys = 2 * out_y + kernel_y[:, pin] - 1
                xs = 2 * out_x + kernel_x[:, pin] - 1
                valid = (ys >= 0) & (ys < height) & (xs >= 0) & (xs < width)
                rows = np.nonzero(valid)[0]
                if rows.size:
                    values[:, rows, pin] = inputs[
                        :, channels[rows, pin], ys[rows], xs[rows]
                    ]
            address = (values << np.arange(6, dtype=np.uint8)).sum(-1)
            output[:, :, out_y, out_x] = truth[np.arange(out_channels)[None, :], address]
    return output


def evaluate_discrete(state: dict, encoded: np.ndarray) -> np.ndarray:
    """Evaluate raw class counts from encoded Boolean CHW input."""
    encoded = np.asarray(encoded)
    if encoded.ndim != 4 or tuple(encoded.shape[1:]) != INPUT_SHAPE:
        raise ValueError(f"Encoded inputs must have shape [N,{','.join(map(str, INPUT_SHAPE))}].")
    if not np.isin(encoded, (0, 1)).all():
        raise ValueError("Encoded circuit inputs must be Boolean.")
    conv1 = _conv_eval(encoded.astype(np.uint8), state["conv1"])
    conv2 = _conv_eval(conv1, state["conv2"])
    dense1 = _dense_eval(conv2.reshape(len(encoded), -1), state["connections"], state["truth_tables"])
    dense2 = _dense_eval(dense1, state["dense2"]["connections"], state["dense2"]["truth_tables"])
    return dense2.reshape(len(encoded), 10, 800).sum(-1, dtype=np.uint16)


def _verification_data(state, model, *, samples: int, seed: int):
    if samples < 1:
        raise ValueError("verify_samples must be positive.")
    if model is None:
        rng = np.random.default_rng(seed)
        encoded = rng.integers(0, 2, size=(samples, *INPUT_SHAPE), dtype=np.uint8)
        if samples >= 1:
            encoded[0] = 0
        if samples >= 2:
            encoded[1] = 1
        expected = evaluate_discrete(state, encoded)
        reference = "validated discrete checkpoint"
    else:
        generator = torch.Generator().manual_seed(seed)
        rgb = torch.rand(samples, 3, 32, 32, generator=generator)
        if samples >= 1:
            rgb[0].zero_()
        if samples >= 2:
            rgb[1].fill_(1)
        with torch.inference_mode():
            encoded_tensor = model.encoder(rgb)
            logits = model(rgb)
        encoded = encoded_tensor.to(torch.uint8).numpy()
        scaled = logits * state["student_tau"]
        expected = scaled.round().to(torch.uint16).numpy()
        torch.testing.assert_close(scaled, scaled.round(), rtol=0, atol=1e-5)
        reference = "best_student hard-forward model"
    actual = evaluate_discrete(state, encoded)
    if not np.array_equal(actual, expected):
        difference = int(np.abs(actual.astype(np.int32) - expected.astype(np.int32)).max())
        raise ValueError(f"Discrete student verification failed; max count difference={difference}.")
    return encoded, expected, reference


def _input_expression(source: str, channel: int, y: int, x: int, height: int, width: int):
    if y < 0 or y >= height or x < 0 or x >= width:
        return "1'b0"
    index = (channel * height + y) * width + x
    return f"{source}[{index}]"


def _simple_lut_expression(inputs, init):
    """Return an exact constant/projection expression, or None for a real LUT6."""
    full = (1 << 64) - 1
    if init == 0:
        return "1'b0"
    if init == full:
        return "1'b1"
    for pin, expression in enumerate(inputs):
        mask = sum(1 << address for address in range(64) if address & (1 << pin))
        if init == mask:
            return expression
        if init == (full ^ mask):
            return f"~({expression})"
    return None


def _emit_lut(lines, *, name, output, inputs, init):
    expression = _simple_lut_expression(inputs, init)
    if expression is not None:
        lines.append(f"    assign {output} = {expression};")
        return 0
    lines.append(f"    LUT6 #(.INIT(64'h{init:016X})) {name} (")
    lines.append("        " + ", ".join(
        [f".I{pin}({value})" for pin, value in enumerate(inputs)] + [f".O({output})"]
    ) + ");")
    return 1


def _emit_conv(lines, *, label, source, source_shape, layer):
    in_channels, height, width = source_shape
    out_channels = int(layer["out_channels"])
    out_height, out_width = (height + 1) // 2, (width + 1) // 2
    count = out_channels * out_height * out_width
    lines.append(f"    wire [{count - 1}:0] {label};")
    connections = _as_numpy(layer["connections"], np.int64)
    init_values = _init_values(layer["truth_tables"])
    emitted_luts = 0
    for channel in range(out_channels):
        for out_y in range(out_height):
            for out_x in range(out_width):
                inputs = []
                for connection in connections[channel]:
                    source_channel = int(connection // 9)
                    remainder = int(connection % 9)
                    y = 2 * out_y + remainder // 3 - 1
                    x = 2 * out_x + remainder % 3 - 1
                    inputs.append(_input_expression(source, source_channel, y, x, height, width))
                output_index = (channel * out_height + out_y) * out_width + out_x
                emitted_luts += _emit_lut(
                    lines, name=f"u_{label}_c{channel}_y{out_y}_x{out_x}",
                    output=f"{label}[{output_index}]", inputs=inputs,
                    init=init_values[channel],
                )
    lines.append("")
    return (out_channels, out_height, out_width), emitted_luts


def _emit_dense(lines, *, label, source, connections, tables):
    connections = _as_numpy(connections, np.int64)
    init_values = _init_values(tables)
    lines.append(f"    wire [{connections.shape[0] - 1}:0] {label};")
    emitted_luts = 0
    for node, selected in enumerate(connections):
        emitted_luts += _emit_lut(
            lines, name=f"u_{label}_{node}", output=f"{label}[{node}]",
            inputs=[f"{source}[{int(index)}]" for index in selected],
            init=init_values[node],
        )
    lines.append("")
    return emitted_luts


def _emit_groupsum(lines):
    def zero_extend(signal, source_width, target_width):
        padding = target_width - source_width
        if padding == 0:
            return signal
        if padding == 1:
            return f"{{1'b0, {signal}}}"
        return f"{{{{{padding}{{1'b0}}}}, {signal}}}"

    lines.extend([
        "    function [7:0] popcount128;",
        "        input [127:0] value;",
        "        integer i;",
        "        begin",
        "            popcount128 = 8'd0;",
        "            for (i = 0; i < 128; i = i + 1)",
        "                popcount128 = popcount128 + value[i];",
        "        end",
        "    endfunction",
        "",
    ])
    for class_index in range(10):
        partials = []
        start = class_index * 800
        for chunk_index, offset in enumerate(range(0, 800, 128)):
            indices = list(range(start + offset, min(start + offset + 128, start + 800)))
            padding = 128 - len(indices)
            values = [f"dense2[{index}]" for index in reversed(indices)]
            if padding:
                values.insert(0, f"{padding}'b0")
            bits = f"gs_{class_index}_bits_{chunk_index}"
            partial = f"gs_{class_index}_p_{chunk_index}"
            lines.append(f"    wire [127:0] {bits} = {{{', '.join(values)}}};")
            lines.append(f"    wire [7:0] {partial} = popcount128({bits});")
            partials.append((partial, 8))
        level = 0
        while len(partials) > 1:
            combined = []
            for pair in range(0, len(partials), 2):
                left, left_width = partials[pair]
                name = f"gs_{class_index}_l{level}_{pair // 2}"
                if pair + 1 < len(partials):
                    right, right_width = partials[pair + 1]
                    width = max(left_width, right_width) + 1
                    lines.append(
                        f"    wire [{width - 1}:0] {name} = "
                        f"{zero_extend(left, left_width, width)} + "
                        f"{zero_extend(right, right_width, width)};"
                    )
                else:
                    width = left_width + 1
                    lines.append(
                        f"    wire [{width - 1}:0] {name} = {{1'b0, {left}}};"
                    )
                combined.append((name, width))
            partials = combined
            level += 1
        score, width = partials[0]
        lines.append(
            f"    assign scores_flat[{class_index * SCORE_BITS} +: {SCORE_BITS}] = "
            f"{{{{{SCORE_BITS - width}{{1'b0}}}}, {score}}};"
        )
        lines.append("")


def generate_verilog(state: dict, *, return_stats=False):
    raw = build_native_lut6_circuit(state)
    circuit = raw.simplified()
    verilog = circuit.get_verilog_code()
    if not return_stats:
        return verilog
    layers = ("conv1", "conv2", "dense1", "dense2")
    before_counts = raw.gate_counts_by_layer()
    after_counts = circuit.gate_counts_by_layer()
    before_by_layer = {layer: before_counts.get(layer, 0) for layer in layers}
    after_by_layer = {layer: after_counts.get(layer, 0) for layer in layers}
    return verilog, {
        "before_by_layer": before_by_layer,
        "after_by_layer": after_by_layer,
        "before": len(raw.gates),
        "after": len(circuit.gates),
    }


def _c_array_2d(name, values, c_type="uint16_t"):
    values = np.asarray(values)
    rows = ["{" + ",".join(map(str, row.tolist())) + "}" for row in values]
    return f"static const {c_type} {name}[{values.shape[0]}][{values.shape[1]}] = {{\n" + ",\n".join(rows) + "\n};\n"


def _c_u64_array(name, values):
    body = ",\n".join(f"UINT64_C(0x{value:016X})" for value in values)
    return f"static const uint64_t {name}[{len(values)}] = {{\n{body}\n}};\n"


def generate_c(state: dict) -> str:
    return build_native_lut6_circuit(state).simplified().get_c_code()


def _json_circuit(state, checkpoint):
    return build_native_lut6_circuit(state, checkpoint=checkpoint).simplified().to_dict()


def build_native_lut6_circuit(state: dict, *, checkpoint=None) -> NativeLUT6Circuit:
    """Build the spatially expanded, unsimplified student LUT6 circuit."""
    gates = []
    next_id = int(np.prod(INPUT_SHAPE))

    def add_gate(inputs, init, *, layer, name):
        nonlocal next_id
        signal = LUT6Signal(next_id)
        gates.append(LUT6Gate(
            next_id, tuple(inputs), int(init), layer=layer, name=name,
        ))
        next_id += 1
        return signal

    def build_conv(source, source_shape, layer, *, label):
        in_channels, height, width = source_shape
        del in_channels
        connections = _as_numpy(layer["connections"], np.int64)
        init_values = _init_values(layer["truth_tables"])
        out_channels = connections.shape[0]
        out_height, out_width = (height + 1) // 2, (width + 1) // 2
        outputs = []
        for channel in range(out_channels):
            for out_y in range(out_height):
                for out_x in range(out_width):
                    inputs = []
                    for connection in connections[channel]:
                        source_channel = int(connection // 9)
                        remainder = int(connection % 9)
                        y = 2 * out_y + remainder // 3 - 1
                        x = 2 * out_x + remainder % 3 - 1
                        if y < 0 or y >= height or x < 0 or x >= width:
                            inputs.append(LUT6Signal(-1))
                        else:
                            index = (source_channel * height + y) * width + x
                            inputs.append(source[index])
                    outputs.append(add_gate(
                        inputs, init_values[channel], layer=label,
                        name=f"u_{label}_c{channel}_y{out_y}_x{out_x}",
                    ))
        return outputs, (out_channels, out_height, out_width)

    def build_dense(source, connections, tables, *, label):
        selected_inputs = _as_numpy(connections, np.int64)
        init_values = _init_values(tables)
        return [add_gate(
            [source[int(index)] for index in selected], init_values[node],
            layer=label, name=f"u_{label}_{node}",
        ) for node, selected in enumerate(selected_inputs)]

    inputs = [LUT6Signal(index) for index in range(int(np.prod(INPUT_SHAPE)))]
    conv1, shape1 = build_conv(inputs, INPUT_SHAPE, state["conv1"], label="conv1")
    conv2, shape2 = build_conv(conv1, shape1, state["conv2"], label="conv2")
    if shape2 != (64, 8, 8):
        raise ValueError("Unexpected Conv2 output shape.")
    dense1 = build_dense(
        conv2, state["connections"], state["truth_tables"], label="dense1",
    )
    dense2 = build_dense(
        dense1, state["dense2"]["connections"], state["dense2"]["truth_tables"],
        label="dense2",
    )
    sums = [LUT6SumReduction(tuple(dense2[index * 800:(index + 1) * 800]))
            for index in range(10)]
    encoder_state = {
        key: _as_numpy(value).tolist() for key, value in state["encoder"]["state"].items()
    }
    input_metadata = {
        "kind": "encoded Boolean bits, NOT raw RGB",
        "shape": list(INPUT_SHAPE),
        "layout": "CHW, row-major; flattened index=(channel*height+y)*width+x",
        "encoder_configuration": {
            "mode": "hybrid_lbp", "num_quantile_bits": 2, "num_lbp_bits": 2,
        },
        "encoder_state": encoder_state,
        "channel_order": "per RGB channel: quantile bits, then LBP horizontal, vertical",
        "lbp": {
            "radius": 1, "padding": "replicate",
            "comparisons": ["right >= left", "down >= up"],
        },
    }
    output_metadata = {
        "kind": "raw integer class counts", "shape": [10],
        "classes": 10, "group_size": 800,
        "original_logits": "(counts + beta) / tau",
        "tau": state["student_tau"], "beta": 0.0,
    }
    metadata = {
        "checkpoint": str(Path(checkpoint).expanduser().resolve()) if checkpoint else None,
        "student_model": state["student_model"],
        "input": input_metadata,
        "output": output_metadata,
    }
    return NativeLUT6Circuit(INPUT_SHAPE, gates, sums, metadata=metadata, score_bits=SCORE_BITS)


def _verify_native_circuit(circuit, encoded, expected):
    actual = circuit(encoded)
    if not np.array_equal(actual, expected):
        difference = int(np.abs(actual.astype(np.int32) - expected.astype(np.int32)).max())
        raise ValueError(f"Native LUT6 circuit verification failed; max count difference={difference}.")
    return {"samples": len(encoded), "exact_class_counts": True, "same_argmax": True}
    return {
        "format": "torchlogix_lut6_student_circuit_v1",
        "checkpoint": str(Path(checkpoint).expanduser().resolve()),
        "input": {
            "kind": "encoded Boolean bits, NOT raw RGB", "shape": list(INPUT_SHAPE),
            "layout": "CHW row-major", "encoder": {
                "mode": "hybrid_lbp", "quantile_bits": 2, "lbp_bits": 2,
                "state": encoder_state,
            },
        },
        "layers": [
            layer("lut6_conv2d", state["conv1"]["connections"], state["conv1"]["truth_tables"],
                  name="conv1", input_shape=list(INPUT_SHAPE), output_shape=[32, 16, 16],
                  kernel_size=3, stride=2, padding=1, padding_value=0),
            layer("lut6_conv2d", state["conv2"]["connections"], state["conv2"]["truth_tables"],
                  name="conv2", input_shape=[32, 16, 16], output_shape=[64, 8, 8],
                  kernel_size=3, stride=2, padding=1, padding_value=0),
            layer("lut6_dense", state["connections"], state["truth_tables"],
                  name="dense1", input_width=4096, output_width=8000),
            layer("lut6_dense", state["dense2"]["connections"], state["dense2"]["truth_tables"],
                  name="dense2", input_width=8000, output_width=8000),
        ],
        "output": {
            "kind": "raw integer class counts", "classes": 10, "group_size": 800,
            "original_logits": "counts / student_tau", "student_tau": state["student_tau"],
        },
    }


def export_student_checkpoint(checkpoint, output_dir, *, verify_samples=4, seed=0,
                              compile_c=False, c_opt_level=0):
    output_dir = Path(output_dir).expanduser().resolve()
    if c_opt_level not in (0, 1, 2, 3):
        raise ValueError("C optimization level must be 0, 1, 2 or 3.")
    state, metadata, reference_model = load_student_checkpoint(checkpoint)
    encoded, expected, reference = _verification_data(
        state, reference_model, samples=verify_samples, seed=seed,
    )
    raw_circuit = build_native_lut6_circuit(state, checkpoint=checkpoint)
    checks = {
        "before_simplification": _verify_native_circuit(raw_circuit, encoded, expected),
    }
    circuit = raw_circuit.simplified()
    checks["after_simplification"] = _verify_native_circuit(circuit, encoded, expected)
    verilog = circuit.get_verilog_code()
    c_code = circuit.get_c_code()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "circuit.v").write_text(verilog)
    (output_dir / "circuit.c").write_text(c_code)
    circuit.write_json(output_dir / "circuit.json")
    reloaded = NativeLUT6Circuit.from_json_file(output_dir / "circuit.json")
    checks["json_roundtrip"] = _verify_native_circuit(reloaded, encoded, expected)
    if compile_c:
        compiled = circuit.evaluate_compiled(encoded, opt_level=c_opt_level)
        if not np.array_equal(compiled, expected):
            difference = int(np.abs(compiled.astype(np.int32) - expected.astype(np.int32)).max())
            raise ValueError(f"Compiled native LUT6 C verification failed; max count difference={difference}.")
        checks["compiled_c"] = {
            "samples": len(encoded), "exact_class_counts": True, "same_argmax": True,
            "opt_level": c_opt_level,
        }
    np.savez_compressed(output_dir / "verification_vectors.npz",
                        encoded=encoded, class_counts=expected)
    before_by_layer = raw_circuit.gate_counts_by_layer()
    after_by_layer = circuit.gate_counts_by_layer()
    layers = ("conv1", "conv2", "dense1", "dense2")
    before_by_layer = {layer: before_by_layer.get(layer, 0) for layer in layers}
    after_by_layer = {layer: after_by_layer.get(layer, 0) for layer in layers}
    offsets = [reduction.offset for reduction in circuit.sums]
    remaining_inputs = [len(reduction.signals) for reduction in circuit.sums]
    report = {
        "checkpoint": str(Path(checkpoint).expanduser().resolve()),
        **metadata,
        "student_model": state["student_model"],
        "input": circuit.metadata["input"],
        "output": circuit.metadata["output"],
        "gates_before_simplification": len(raw_circuit.gates),
        "gates_after_simplification": len(circuit.gates),
        "gate_count_note": "Spatially expanded native LUT6 nodes after iterative constant/alias propagation, LUT cofactoring, deduplication and dead-node elimination. Encoder and GroupSum arithmetic excluded.",
        "native_lut6_instances": len(circuit.gates),
        "layer_lut6_evaluations_before_simplification": before_by_layer,
        "layer_lut6_instances_after_simplification": after_by_layer,
        "groupsum": {"classes": 10, "inputs_per_class": 800,
                     "remaining_inputs_per_class": remaining_inputs,
                     "folded_constant_offsets": offsets,
                     "implementation": "128-bit local popcounts plus balanced partial-sum trees"},
        "verification": {**checks, "reference": reference},
        "warnings": [],
        "rtl_note": "Xilinx native LUT6 primitives; combinational encoded-input core only. No RGB encoder, clock, pipeline, or board I/O.",
    }
    (output_dir / "export_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="discrete_student.pt or best_student.pt for tinyconv_q2_lbp2_dense2")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--verify-samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--compile-c", action="store_true",
                        help="Compile generated C with GCC and verify exact class counts.")
    parser.add_argument("--c-opt-level", type=int, choices=(0, 1, 2, 3), default=0)
    args = parser.parse_args()
    try:
        report = export_student_checkpoint(
            args.checkpoint, args.output_dir,
            verify_samples=args.verify_samples, seed=args.seed,
            compile_c=args.compile_c, c_opt_level=args.c_opt_level,
        )
    except (ValueError, RuntimeError, OSError, KeyError, AssertionError) as error:
        parser.exit(1, f"Error: {error}\n")
    print(f"Exported {report['gates_before_simplification']:,} -> "
          f"{report['gates_after_simplification']:,} native LUT6 instances.")
    print("Saved circuit.json, circuit.c, circuit.v, export_report.json and verification_vectors.npz "
          f"in {args.output_dir.resolve()}")
    print("Inputs are encoded Q2+LBP2 bits; outputs are ten raw integer class counts.")


if __name__ == "__main__":
    main()
