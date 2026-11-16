"""Shared CompactLogic heads without MRLB or class-local routing."""

import math

import torch

from torchlogix.layers import GroupSum

from .classrouted_compact import (
    build_model_cifar10_small_classrouted_compact_head,
)
from .compact import CandidateLogicLayer
from .heartbeat import HeartbeatQ2LocalEncoder


def _build_plain_shared_compact(
    device, encoder, input_shape, *, compact_backend, dense_width, num_classes,
):
    shape = tuple(input_shape)
    layers = [encoder]
    for channels, stride in ((128, 2), (128, 1), (512, 2), (512, 1)):
        layers.append(CandidateLogicLayer(
            device=device, backend=compact_backend, in_dim=math.prod(shape),
            out_dim=channels, in_shape=shape, stride=stride,
        ))
        shape = (channels, (shape[1] + stride - 1) // stride,
                 (shape[2] + stride - 1) // stride)
    layers.append(torch.nn.Flatten())
    layers.append(CandidateLogicLayer(
        device=device, backend=compact_backend,
        in_dim=math.prod(shape), out_dim=dense_width,
    ))
    layers.append(CandidateLogicLayer(
        device=device, backend=compact_backend, in_dim=dense_width, out_dim=dense_width,
    ))
    return torch.nn.Sequential(*layers, GroupSum(k=num_classes, tau=55)).to(device)


def build_model_cifar10_small_compact(
    device, *, parametrization="compact", input_thresholds=None,
    input_encoding="hybrid", num_quantile_bits=2, num_sine_bits=2,
    sine_sigma=1.0, sine_seed=0, compact_backend="auto", dense_width=92_400,
):
    """Plain four-conv model with a shared dense width (92.4k by default)."""
    from .model import _build_cifar10_encoder

    if parametrization != "compact":
        raise ValueError("The no-skip candidate head requires --parametrization compact.")
    encoder = _build_cifar10_encoder(
        input_thresholds,
        input_encoding=input_encoding, num_quantile_bits=num_quantile_bits,
        num_sine_bits=num_sine_bits, sine_sigma=sine_sigma, sine_seed=sine_seed,
    )
    return _build_plain_shared_compact(
        device, encoder, (encoder.output_channels, 32, 32),
        compact_backend=compact_backend, dense_width=dense_width, num_classes=10,
    )


def build_model_cifar100_small_compact(
    device, *, parametrization="compact", input_thresholds=None,
    input_encoding="hybrid", num_quantile_bits=2, num_sine_bits=2,
    sine_sigma=1.0, sine_seed=0, compact_backend="auto", dense_width=92_400,
):
    """Plain CIFAR-100 model with the CIFAR-10 backbone and shared dense head."""
    from .model import _build_cifar10_encoder

    if parametrization != "compact":
        raise ValueError("The plain shared CompactLogic model requires compact parametrization.")
    encoder = _build_cifar10_encoder(
        input_thresholds, input_encoding=input_encoding,
        num_quantile_bits=num_quantile_bits, num_sine_bits=num_sine_bits,
        sine_sigma=sine_sigma, sine_seed=sine_seed,
    )
    return _build_plain_shared_compact(
        device, encoder, (encoder.output_channels, 32, 32),
        compact_backend=compact_backend, dense_width=dense_width, num_classes=100,
    )


def build_model_mnist_small_compact(
    device, *, parametrization="compact", compact_backend="auto", dense_width=92_400,
):
    """Plain MNIST model with a binary grayscale input and shared dense head."""
    if parametrization != "compact":
        raise ValueError("The plain shared CompactLogic model requires compact parametrization.")
    return _build_plain_shared_compact(
        device, torch.nn.Identity(), (1, 28, 28),
        compact_backend=compact_backend, dense_width=dense_width, num_classes=10,
    )


def build_model_heartbeat_small_compact(
    device, *, parametrization="compact", input_thresholds=None,
    input_encoding="hybrid", num_quantile_bits=2, num_sine_bits=0,
    compact_backend="auto", dense_width=92_400,
):
    """Four 128/128/512/512 convolutions, two shared dense layers, five scores."""
    if parametrization != "compact":
        raise ValueError("The heartbeat model requires compact parametrization.")
    if input_encoding != "hybrid" or num_quantile_bits != 2 or num_sine_bits != 0:
        raise ValueError("The heartbeat model requires hybrid Q2 + local comparisons.")
    encoder = HeartbeatQ2LocalEncoder(input_thresholds)
    return _build_plain_shared_compact(
        device, encoder, (encoder.output_channels, 11, 17),
        compact_backend=compact_backend, dense_width=dense_width, num_classes=5,
    )


def build_model_cifar10_small_compact_88000(
    device, *, parametrization="compact", input_thresholds=None,
    input_encoding="hybrid", num_quantile_bits=2, num_sine_bits=2,
    sine_sigma=1.0, sine_seed=0, compact_backend="auto",
):
    """The same plain four-conv model with two 88,000-output dense layers."""
    return build_model_cifar10_small_compact(
        device, parametrization=parametrization, input_thresholds=input_thresholds,
        input_encoding=input_encoding, num_quantile_bits=num_quantile_bits,
        num_sine_bits=num_sine_bits, sine_sigma=sine_sigma, sine_seed=sine_seed,
        compact_backend=compact_backend, dense_width=88_000,
    )


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
