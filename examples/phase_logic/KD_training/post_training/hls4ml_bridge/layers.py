"""Custom hls4ml IR layers for fixed LUT6 inference."""

from __future__ import annotations

import hls4ml
from hls4ml.model.layers import Layer
from hls4ml.model.types import IntegerPrecisionType


BOOL_T = IntegerPrecisionType(width=1, signed=False)
COUNT_T = IntegerPrecisionType(width=10, signed=False)


class LUT6Conv2D(Layer):
    def initialize(self):
        self.add_output_variable(self.get_attr("output_shape"), precision=BOOL_T)


class LUT6Dense(Layer):
    def initialize(self):
        self.add_output_variable(self.get_attr("output_shape"), precision=BOOL_T)


class BooleanGroupSum(Layer):
    def initialize(self):
        if self.get_attr("group_size") > (1 << COUNT_T.width) - 1:
            raise ValueError("BooleanGroupSum does not fit in the required unsigned 10-bit output.")
        self.add_output_variable(self.get_attr("output_shape"), precision=COUNT_T)


def register_layers():
    """Register custom nodes with hls4ml's internal layer registry."""
    registry = hls4ml.model.layers.layer_map
    for name, layer in (
        ("LUT6Conv2D", LUT6Conv2D),
        ("LUT6Dense", LUT6Dense),
        ("BooleanGroupSum", BooleanGroupSum),
    ):
        registered = registry.get(name)
        if registered is None:
            hls4ml.model.layers.register_layer(name, layer)
        elif registered is not layer:
            raise RuntimeError(f"hls4ml layer name {name} is already registered by another implementation.")

