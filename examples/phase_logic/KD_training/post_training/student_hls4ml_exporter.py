"""Export the two-dense LUT6 student as a layer-oriented hls4ml custom IR.

This JSON is an input for a future custom hls4ml frontend. It is not a format
that stock hls4ml can consume directly. The inference interface is the
pre-encoded Boolean tensor, not an RGB image.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .load_checkpoint import load_student_checkpoint
from .student_circuit_exporter import _as_numpy, _init_values


FORMAT = "hls4ml_lut6_ir_v1"


def _json_value(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _output_size(size: int, kernel: int, stride: int, padding: int) -> int:
    result = (size + 2 * padding - kernel) // stride + 1
    if result <= 0:
        raise ValueError("Convolution geometry produces a nonpositive output size.")
    return result


def _validate_lut6(name: str, connections, truth_tables, *, nodes: int, input_width: int):
    connections = _as_numpy(connections, np.int64)
    truth_tables = _as_numpy(truth_tables, np.bool_)
    if connections.shape != (nodes, 6):
        raise ValueError(f"{name} connections must have shape ({nodes}, 6).")
    if truth_tables.shape != (nodes, 64):
        raise ValueError(f"{name} truth tables must have shape ({nodes}, 64).")
    if connections.size and (connections.min() < 0 or connections.max() >= input_width):
        raise ValueError(f"{name} contains an out-of-range connection index.")
    return connections, truth_tables


def _init_hex(truth_tables) -> list[str]:
    return [f"{value:016X}" for value in _init_values(truth_tables)]


def _conv_layer(name: str, input_name: str, input_shape, layer: dict):
    in_channels, height, width = map(int, input_shape)
    out_channels = int(layer["out_channels"])
    kernel = int(layer["kernel_size"])
    stride = int(layer["stride"])
    padding = int(layer["padding"])
    if int(layer["in_channels"]) != in_channels:
        raise ValueError(f"{name} input channels do not match the preceding tensor.")
    local_patch_width = in_channels * kernel * kernel
    connections, truth_tables = _validate_lut6(
        name, layer["connections"], layer["truth_tables"],
        nodes=out_channels, input_width=local_patch_width,
    )
    output_shape = [
        out_channels,
        _output_size(height, kernel, stride, padding),
        _output_size(width, kernel, stride, padding),
    ]
    return {
        "name": name,
        "op_type": "LUT6Conv2D",
        "input": input_name,
        "input_shape": [in_channels, height, width],
        "output_shape": output_shape,
        "kernel_size": [kernel, kernel],
        "stride": [stride, stride],
        "padding": [padding, padding],
        "padding_value": int(layer.get("padding_value", 0)),
        "local_patch_flattening": {
            "order": "channel,row,column",
            "index_formula": "channel * (kernel_height * kernel_width) + row * kernel_width + column",
        },
        "connections": connections.tolist(),
        "init_hex": _init_hex(truth_tables),
        "pin0_is_address_lsb": True,
        "weight_sharing": {
            "semantics": "one connection vector and one truth table per output channel",
            "shared_across": ["output_row", "output_column"],
            "parameter_index": "output_channel",
        },
    }


def _dense_layer(name: str, input_name: str, input_width: int, connections, truth_tables):
    node_count = int(_as_numpy(connections).shape[0])
    connections, truth_tables = _validate_lut6(
        name, connections, truth_tables, nodes=node_count, input_width=input_width,
    )
    return {
        "name": name,
        "op_type": "LUT6Dense",
        "input": input_name,
        "input_shape": [input_width],
        "output_shape": [node_count],
        "connections": connections.tolist(),
        "init_hex": _init_hex(truth_tables),
        "pin0_is_address_lsb": True,
    }


def build_hls4ml_ir(state: dict, metadata: dict, checkpoint) -> dict:
    """Transform a validated discrete student state into hls4ml_lut6_ir_v1."""
    input_shape = [int(state["conv1"]["in_channels"]), 32, 32]
    conv1 = _conv_layer("conv1", "encoded_input", input_shape, state["conv1"])
    conv2 = _conv_layer("conv2", "conv1", conv1["output_shape"], state["conv2"])

    flattened_width = int(np.prod(conv2["output_shape"]))
    declared_head_shape = tuple(int(value) for value in state["head_input_shape"])
    if tuple(conv2["output_shape"]) != declared_head_shape:
        raise ValueError("Conv2 output shape does not match the Dense1 input shape.")
    dense1 = _dense_layer(
        "dense1", "conv2", flattened_width,
        state["connections"], state["truth_tables"],
    )
    dense2 = _dense_layer(
        "dense2", "dense1", dense1["output_shape"][0],
        state["dense2"]["connections"], state["dense2"]["truth_tables"],
    )

    group_size = int(state["group_size"])
    dense2_width = dense2["output_shape"][0]
    if group_size <= 0 or dense2_width % group_size:
        raise ValueError("Dense2 width must divide exactly into equal GroupSum classes.")
    classes = dense2_width // group_size
    if classes != 10 or state.get("groupsum_source") != "dense2":
        raise ValueError("The supported architecture requires ten Dense2 GroupSum classes.")
    class_groups = [
        {
            "class_index": index,
            "node_start": index * group_size,
            "node_stop_exclusive": (index + 1) * group_size,
        }
        for index in range(classes)
    ]
    group_sum = {
        "name": "group_sum",
        "op_type": "BooleanGroupSum",
        "input": "dense2",
        "input_shape": [dense2_width],
        "output_shape": [classes],
        "classes": classes,
        "group_size": group_size,
        "class_groups": class_groups,
        "raw_output": {
            "name": "class_counts",
            "dtype": "unsigned_integer",
            "range_inclusive": [0, group_size],
            "semantics": "count TRUE nodes in each class group",
        },
        "original_tau": float(state["student_tau"]),
        "scaled_logit_interpretation": "class_counts[class_index] / original_tau",
    }

    encoder = state["encoder"]
    if int(encoder["lbp_bits"]):
        channel_order = (
            "per RGB channel: quantile bits, LBP horizontal, LBP vertical"
            if int(encoder["quantile_bits"]) else
            "per RGB channel: LBP horizontal, LBP vertical"
        )
        lbp_metadata = {
            "radius": 1,
            "padding": "replicate",
            "comparisons": ["right >= left", "down >= up"],
        }
    else:
        channel_order = "per RGB channel: quantile bits"
        lbp_metadata = None
    layers = [conv1, conv2, dense1, dense2, group_sum]
    return {
        "format": FORMAT,
        "source": {
            "checkpoint": str(Path(checkpoint).expanduser().resolve()),
            "checkpoint_kind": metadata.get("checkpoint_kind"),
            "epoch": metadata.get("epoch"),
            "validation": _json_value(metadata.get("validation")),
            "student_model": state["student_model"],
        },
        "input": {
            "name": "encoded_input",
            "dtype": "boolean",
            "shape": input_shape,
            "layout": "CHW",
            "flattening": {
                "order": "channel,row,column",
                "index_formula": "(channel * height + row) * width + column",
            },
            "interface": "pre-encoded Boolean bits; raw RGB encoding is outside this IR",
            "encoder_metadata": {
                "mode": encoder["mode"],
                "quantile_bits": int(encoder["quantile_bits"]),
                "lbp_bits": int(encoder["lbp_bits"]),
                "state": _json_value(encoder["state"]),
                "channel_order": channel_order,
                "lbp": lbp_metadata,
            },
        },
        "graph": {
            "inputs": ["encoded_input"],
            "outputs": ["class_counts"],
            "execution_order": [layer["name"] for layer in layers],
            "edges": [
                ["encoded_input", "conv1"],
                ["conv1", "conv2"],
                ["conv2", "dense1"],
                ["dense1", "dense2"],
                ["dense2", "group_sum"],
                ["group_sum", "class_counts"],
            ],
        },
        "layers": layers,
        "outputs": [
            {
                "name": "class_counts",
                "source": "group_sum",
                "dtype": "unsigned_integer",
                "shape": [classes],
                "semantics": "exact raw integer class counts; prediction is argmax(class_counts)",
            }
        ],
    }


def export_hls4ml_ir(checkpoint, output):
    state, metadata, _ = load_student_checkpoint(checkpoint)
    document = build_hls4ml_ir(state, metadata, checkpoint)
    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2) + "\n")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, required=True,
        help="best_student.pt or discrete_student.pt for a supported two-dense LUT6 student",
    )
    parser.add_argument("--output", type=Path, required=True, help="Destination JSON file.")
    args = parser.parse_args()
    try:
        output = export_hls4ml_ir(args.checkpoint, args.output)
    except (ValueError, RuntimeError, OSError, KeyError, TypeError) as error:
        parser.exit(1, f"Error: {error}\n")
    print(f"Saved {FORMAT} to {output}")


if __name__ == "__main__":
    main()
