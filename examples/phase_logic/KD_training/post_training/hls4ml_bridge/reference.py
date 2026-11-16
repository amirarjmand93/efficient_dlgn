"""Validation and exact NumPy inference for hls4ml_lut6_ir_v1."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


FORMAT = "hls4ml_lut6_ir_v1"


def _positive_shape(value, label):
    if not isinstance(value, list) or not value or any(not isinstance(x, int) or x <= 0 for x in value):
        raise ValueError(f"{label} must be a nonempty list of positive integers.")
    return tuple(value)


def _validate_lut_layer(layer, input_width):
    output_shape = _positive_shape(layer.get("output_shape"), f"{layer['name']}.output_shape")
    nodes = output_shape[0] if layer["op_type"] == "LUT6Conv2D" else output_shape[-1]
    connections = np.asarray(layer.get("connections"))
    tables = layer.get("init_hex")
    if connections.shape != (nodes, 6):
        raise ValueError(f"{layer['name']} connections must have shape ({nodes}, 6).")
    if not np.issubdtype(connections.dtype, np.integer):
        raise ValueError(f"{layer['name']} connections must be integers.")
    if connections.size and (connections.min() < 0 or connections.max() >= input_width):
        raise ValueError(f"{layer['name']} has an out-of-range connection.")
    if not isinstance(tables, list) or len(tables) != nodes:
        raise ValueError(f"{layer['name']} must have one init_hex value per node/filter.")
    for value in tables:
        if not isinstance(value, str) or len(value) != 16:
            raise ValueError(f"{layer['name']} init_hex entries must contain exactly 64 bits.")
        try:
            int(value, 16)
        except ValueError as error:
            raise ValueError(f"{layer['name']} contains invalid hexadecimal truth-table data.") from error
    if layer.get("pin0_is_address_lsb") is not True:
        raise ValueError(f"{layer['name']} must use pin 0 as the LUT address LSB.")


def validate_ir(document):
    if not isinstance(document, dict) or document.get("format") != FORMAT:
        raise ValueError(f"Expected JSON format {FORMAT}.")
    input_spec = document.get("input")
    if not isinstance(input_spec, dict) or input_spec.get("dtype") != "boolean":
        raise ValueError("IR input must be Boolean.")
    input_shape = _positive_shape(input_spec.get("shape"), "input.shape")
    if input_spec.get("layout") != "CHW":
        raise ValueError("Only CHW input layout is supported.")

    layers = document.get("layers")
    if not isinstance(layers, list) or len(layers) != 5:
        raise ValueError("Expected two LUT6Conv2D, two LUT6Dense and one BooleanGroupSum layer.")
    expected_ops = ["LUT6Conv2D", "LUT6Conv2D", "LUT6Dense", "LUT6Dense", "BooleanGroupSum"]
    if [layer.get("op_type") for layer in layers] != expected_ops:
        raise ValueError("Layer operation order does not match the supported student.")
    names = [layer.get("name") for layer in layers]
    if any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(names):
        raise ValueError("Layer names must be nonempty and unique.")

    previous_name = input_spec.get("name")
    previous_shape = input_shape
    for layer in layers[:-1]:
        if layer.get("input") != previous_name:
            raise ValueError(f"{layer['name']} is not connected to {previous_name}.")
        declared_input = _positive_shape(layer.get("input_shape"), f"{layer['name']}.input_shape")
        if layer["op_type"] == "LUT6Dense" and len(previous_shape) > 1:
            expected_input = (int(np.prod(previous_shape)),)
        else:
            expected_input = previous_shape
        if declared_input != expected_input:
            raise ValueError(f"{layer['name']} input shape does not match its source.")
        if layer["op_type"] == "LUT6Conv2D":
            kernel = _positive_shape(layer.get("kernel_size"), f"{layer['name']}.kernel_size")
            stride = _positive_shape(layer.get("stride"), f"{layer['name']}.stride")
            padding = tuple(layer.get("padding", ()))
            if len(kernel) != 2 or len(stride) != 2 or len(padding) != 2 or any(x < 0 for x in padding):
                raise ValueError(f"{layer['name']} has invalid 2D convolution geometry.")
            local_width = declared_input[0] * kernel[0] * kernel[1]
            _validate_lut_layer(layer, local_width)
            calculated = (
                layer["output_shape"][0],
                (declared_input[1] + 2 * padding[0] - kernel[0]) // stride[0] + 1,
                (declared_input[2] + 2 * padding[1] - kernel[1]) // stride[1] + 1,
            )
            if tuple(layer["output_shape"]) != calculated:
                raise ValueError(f"{layer['name']} output shape disagrees with its convolution geometry.")
        else:
            _validate_lut_layer(layer, declared_input[0])
        previous_name = layer["name"]
        previous_shape = tuple(layer["output_shape"])

    group = layers[-1]
    if group.get("input") != previous_name or tuple(group.get("input_shape", ())) != previous_shape:
        raise ValueError("BooleanGroupSum input does not match Dense2 output.")
    classes, group_size = group.get("classes"), group.get("group_size")
    if not isinstance(classes, int) or not isinstance(group_size, int) or classes <= 0 or group_size <= 0:
        raise ValueError("BooleanGroupSum classes and group_size must be positive integers.")
    if classes * group_size != previous_shape[0] or tuple(group.get("output_shape", ())) != (classes,):
        raise ValueError("BooleanGroupSum dimensions are inconsistent.")
    expected_groups = [
        {"class_index": index, "node_start": index * group_size,
         "node_stop_exclusive": (index + 1) * group_size}
        for index in range(classes)
    ]
    if group.get("class_groups") != expected_groups:
        raise ValueError("BooleanGroupSum class-to-node ranges are not contiguous and exact.")
    return document


def load_ir(path):
    with Path(path).expanduser().open() as stream:
        return validate_ir(json.load(stream))


def _tables(layer):
    return np.asarray([int(value, 16) for value in layer["init_hex"]], dtype=np.uint64)


def _lut_values(bits, tables):
    address = np.sum(bits.astype(np.uint8) << np.arange(6, dtype=np.uint8), axis=-1)
    return ((tables[np.newaxis, :] >> address) & np.uint64(1)).astype(np.uint8)


def _conv(inputs, layer):
    connections = np.asarray(layer["connections"], dtype=np.int64)
    tables = _tables(layer)
    batch, in_channels, in_height, in_width = inputs.shape
    out_channels, out_height, out_width = layer["output_shape"]
    kernel_height, kernel_width = layer["kernel_size"]
    stride_height, stride_width = layer["stride"]
    pad_height, pad_width = layer["padding"]
    channels = connections // (kernel_height * kernel_width)
    remainder = connections % (kernel_height * kernel_width)
    kernel_rows, kernel_columns = remainder // kernel_width, remainder % kernel_width
    output = np.empty((batch, out_channels, out_height, out_width), dtype=np.uint8)
    for out_row in range(out_height):
        for out_column in range(out_width):
            selected = np.full(
                (batch, out_channels, 6), int(layer["padding_value"]), dtype=np.uint8,
            )
            rows = out_row * stride_height + kernel_rows - pad_height
            columns = out_column * stride_width + kernel_columns - pad_width
            valid = (rows >= 0) & (rows < in_height) & (columns >= 0) & (columns < in_width)
            filters, pins = np.nonzero(valid)
            selected[:, filters, pins] = inputs[
                :, channels[filters, pins], rows[filters, pins], columns[filters, pins]
            ]
            output[:, :, out_row, out_column] = _lut_values(selected, tables)
    return output


def _dense(inputs, layer):
    inputs = inputs.reshape(len(inputs), -1)
    connections = np.asarray(layer["connections"], dtype=np.int64)
    selected = inputs[:, connections]
    return _lut_values(selected, _tables(layer))


def predict(document, encoded):
    validate_ir(document)
    encoded = np.asarray(encoded)
    expected_shape = tuple(document["input"]["shape"])
    if encoded.ndim != 4 or tuple(encoded.shape[1:]) != expected_shape:
        raise ValueError(f"Encoded input must have shape [batch,{','.join(map(str, expected_shape))}].")
    if not np.isin(encoded, (0, 1)).all():
        raise ValueError("Encoded input values must be Boolean.")
    value = encoded.astype(np.uint8, copy=False)
    for layer in document["layers"]:
        if layer["op_type"] == "LUT6Conv2D":
            value = _conv(value, layer)
        elif layer["op_type"] == "LUT6Dense":
            value = _dense(value, layer)
        else:
            value = value.reshape(len(value), layer["classes"], layer["group_size"]).sum(
                axis=-1, dtype=np.uint16
            )
    return value
