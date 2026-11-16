"""Vivado/Vitis templates for the custom LUT6 hls4ml layers."""

from __future__ import annotations

from pathlib import Path

import hls4ml

from .layers import BooleanGroupSum, LUT6Conv2D, LUT6Dense


HEADER = Path(__file__).with_name("nnet_lut6.h").resolve()
INCLUDE = ["nnet_utils/nnet_lut6.h"]
_REGISTERED = set()
DENSE_BANK_WIDTH = 64
DENSE_TILE_SIZE = 128
GROUP_SUM_TILE_SIZE = 128


def _connection_initializer(connections):
    return "{\n" + ",\n".join(
        "        {" + ", ".join(str(int(value)) for value in row) + "}"
        for row in connections
    ) + "\n    }"


def _truth_initializer(init_hex):
    return "{\n        " + ",\n        ".join(f"0x{value}ULL" for value in init_hex) + "\n    }"


class LUT6Conv2DConfigTemplate(hls4ml.backends.template.LayerConfigTemplate):
    def __init__(self):
        super().__init__(LUT6Conv2D)

    def format(self, node):
        index = node.index
        input_shape = node.get_attr("input_shape")
        output_shape = node.get_attr("output_shape")
        kernel = node.get_attr("kernel_size")
        stride = node.get_attr("stride")
        padding = node.get_attr("padding")
        connections = node.get_attr("connections")
        init_hex = node.get_attr("init_hex")
        connection_bits = max(1, (input_shape[0] * kernel[0] * kernel[1] - 1).bit_length())
        return f"""struct config{index} : nnet::lut6_conv2d_config {{
    static const unsigned in_channels = {input_shape[0]};
    static const unsigned in_height = {input_shape[1]};
    static const unsigned in_width = {input_shape[2]};
    static const unsigned out_channels = {output_shape[0]};
    static const unsigned out_height = {output_shape[1]};
    static const unsigned out_width = {output_shape[2]};
    static const unsigned kernel_height = {kernel[0]};
    static const unsigned kernel_width = {kernel[1]};
    static const unsigned stride_height = {stride[0]};
    static const unsigned stride_width = {stride[1]};
    static const unsigned pad_height = {padding[0]};
    static const unsigned pad_width = {padding[1]};
    static const unsigned padding_value = {node.get_attr('padding_value')};
    static const unsigned n_in = in_channels * in_height * in_width;
    static const unsigned n_out = out_channels * out_height * out_width;
    static const unsigned bank_width = {DENSE_BANK_WIDTH};
    static const unsigned output_banks = (n_out + bank_width - 1) / bank_width;
    static const unsigned banks_per_filter = (out_height * out_width) / bank_width;
    static const ap_uint<{connection_bits}> connections[out_channels][6];
    static const ap_uint<64> truth_tables[out_channels];
}};
const ap_uint<{connection_bits}> config{index}::connections[config{index}::out_channels][6] = {_connection_initializer(connections)};
const ap_uint<64> config{index}::truth_tables[config{index}::out_channels] = {_truth_initializer(init_hex)};
"""


class LUT6DenseConfigTemplate(hls4ml.backends.template.LayerConfigTemplate):
    def __init__(self):
        super().__init__(LUT6Dense)

    def format(self, node):
        index = node.index
        n_in = node.get_attr("input_shape")[0]
        n_out = node.get_attr("output_shape")[0]
        connection_bits = max(1, (n_in - 1).bit_length())
        return f"""struct config{index} : nnet::lut6_dense_config {{
    static const unsigned n_in = {n_in};
    static const unsigned n_out = {n_out};
    static const unsigned bank_width = {DENSE_BANK_WIDTH};
    static const unsigned output_banks = (n_out + bank_width - 1) / bank_width;
    static const unsigned tile_size = {DENSE_TILE_SIZE};
    static const unsigned n_tiles = (n_out + tile_size - 1) / tile_size;
    static const ap_uint<{connection_bits}> connections[n_out][6];
    static const ap_uint<64> truth_tables[n_out];
}};
const ap_uint<{connection_bits}> config{index}::connections[config{index}::n_out][6] = {_connection_initializer(node.get_attr('connections'))};
const ap_uint<64> config{index}::truth_tables[config{index}::n_out] = {_truth_initializer(node.get_attr('init_hex'))};
"""


class BooleanGroupSumConfigTemplate(hls4ml.backends.template.LayerConfigTemplate):
    def __init__(self):
        super().__init__(BooleanGroupSum)

    def format(self, node):
        index = node.index
        return f"""struct config{index} : nnet::boolean_group_sum_config {{
    static const unsigned classes = {node.get_attr('classes')};
    static const unsigned group_size = {node.get_attr('group_size')};
    static const unsigned n_in = classes * group_size;
    static const unsigned bank_width = {DENSE_BANK_WIDTH};
    static const unsigned tile_size = {GROUP_SUM_TILE_SIZE};
    static const unsigned tiles_per_class = (group_size + tile_size - 1) / tile_size;
}};
"""


class LUT6Conv2DFunctionTemplate(hls4ml.backends.template.FunctionCallTemplate):
    def __init__(self):
        super().__init__(LUT6Conv2D, include_header=INCLUDE)

    def format(self, node):
        params = self._default_function_params(node)
        function = "lut6_conv2d_flat" if node.get_attr("input_is_flat") else "lut6_conv2d_packed"
        return f"nnet::{function}<{{input_t}}, {{output_t}}, {{config}}>({{input}}, {{output}});".format(**params)


class LUT6DenseFunctionTemplate(hls4ml.backends.template.FunctionCallTemplate):
    def __init__(self):
        super().__init__(LUT6Dense, include_header=INCLUDE)

    def format(self, node):
        params = self._default_function_params(node)
        return "nnet::lut6_dense_packed<{input_t}, {output_t}, {config}>({input}, {output});".format(**params)


class BooleanGroupSumFunctionTemplate(hls4ml.backends.template.FunctionCallTemplate):
    def __init__(self):
        super().__init__(BooleanGroupSum, include_header=INCLUDE)

    def format(self, node):
        params = self._default_function_params(node)
        return "nnet::boolean_group_sum_packed<{input_t}, {output_t}, {config}>({input}, {output});".format(**params)


def register_backend(backend_name):
    """Register templates and the HLS header with one hls4ml backend."""
    registry_name = backend_name.lower()
    if registry_name in _REGISTERED:
        return
    if registry_name not in hls4ml.backends.get_available_backends():
        raise ValueError(f"hls4ml backend {backend_name!r} is unavailable.")
    backend = hls4ml.backends.get_backend(registry_name)
    for template in (
        LUT6Conv2DConfigTemplate,
        LUT6DenseConfigTemplate,
        BooleanGroupSumConfigTemplate,
        LUT6Conv2DFunctionTemplate,
        LUT6DenseFunctionTemplate,
        BooleanGroupSumFunctionTemplate,
    ):
        backend.register_template(template)
    backend.register_source(HEADER)
    _REGISTERED.add(registry_name)
