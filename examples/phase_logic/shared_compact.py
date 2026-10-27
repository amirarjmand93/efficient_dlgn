"""Shared CompactLogic heads without MRLB or class-local routing."""

from .classrouted_compact import (
    build_model_cifar10_small_classrouted_compact_head,
)
from .compact import CandidateLogicLayer


def build_model_cifar10_small_compact_head(
    device, *, parametrization="compact", input_thresholds=None,
    input_encoding="hybrid", num_quantile_bits=2, num_sine_bits=2,
    sine_sigma=1.0, sine_seed=0, compact_backend="auto",
):
    """Original four-conv skips; two shared dense stages, each 5% wider than 88k."""
    baseline = build_model_cifar10_small_classrouted_compact_head(
        device, parametrization=parametrization, input_thresholds=input_thresholds,
        input_encoding=input_encoding, num_quantile_bits=num_quantile_bits,
        num_sine_bits=num_sine_bits, sine_sigma=sine_sigma, sine_seed=sine_seed,
        compact_backend=compact_backend,
    )
    baseline[6] = CandidateLogicLayer(
        device=device, backend=compact_backend, in_dim=baseline[6].impl.in_dim, out_dim=92_400,
    )
    baseline[7] = CandidateLogicLayer(
        device=device, backend=compact_backend, in_dim=92_400, out_dim=92_400,
    )
    return baseline.to(device)


def build_model_cifar10_small_compact_head_wide75(
    device, *, parametrization="compact", input_thresholds=None,
    input_encoding="hybrid", num_quantile_bits=2, num_sine_bits=2,
    sine_sigma=1.0, sine_seed=0, compact_backend="auto",
):
    """Separate variant with 75% wider convolutions and shared dense stages."""
    model = build_model_cifar10_small_compact_head(
        device, parametrization=parametrization, input_thresholds=input_thresholds,
        input_encoding=input_encoding, num_quantile_bits=num_quantile_bits,
        num_sine_bits=num_sine_bits, sine_sigma=sine_sigma, sine_seed=sine_seed,
        compact_backend=compact_backend,
    )
    shape = (model[0].output_channels, 32, 32)
    for conv_index, (channels, stride) in enumerate(((224, 2), (224, 1), (896, 2), (896, 1))):
        if conv_index == 2:
            shape = (2 * shape[0], *shape[1:])
        model[conv_index + 1] = CandidateLogicLayer(
            device=device, backend=compact_backend,
            in_dim=shape[0] * shape[1] * shape[2], out_dim=channels,
            in_shape=shape, stride=stride,
        )
        shape = (channels, (shape[1] + stride - 1) // stride,
                 (shape[2] + stride - 1) // stride)
    model[6] = CandidateLogicLayer(
        device=device, backend=compact_backend,
        in_dim=2 * shape[0] * shape[1] * shape[2], out_dim=161_700,
    )
    model[7] = CandidateLogicLayer(
        device=device, backend=compact_backend, in_dim=161_700, out_dim=161_700,
    )
    return model.to(device)
