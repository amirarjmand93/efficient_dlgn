"""Strict loader for discrete or trainable two-dense LUT6 student checkpoints."""

from __future__ import annotations

import math
from pathlib import Path

import torch

from ..student import (TinyConv2DenseLUT6Student, TinyConvQ2Wide10Dense2LUT6Student,
                       TinyConvQ2Compact20Dense2LUT6Student,
                       TinyConvLBP2Wide10Dense2LUT6Student)


_SUPPORTED_MODELS = {
    "tinyconv_q2_lbp2_dense2": {
        "class": TinyConv2DenseLUT6Student,
        "encoder_mode": "hybrid_lbp", "encoder_channels": 12,
        "conv_channels": (32, 64), "dense_width": 8000,
        "group_size": 800, "quantile_bits": 2, "lbp_bits": 2,
    },
    "tinyconv_q2_wide10_dense2": {
        "class": TinyConvQ2Wide10Dense2LUT6Student,
        "encoder_mode": "quantile", "encoder_channels": 6,
        "conv_channels": (35, 70), "dense_width": 8800,
        "group_size": 880, "quantile_bits": 2, "lbp_bits": 0,
    },
    "tinyconv_q2_compact20_dense2": {
        "class": TinyConvQ2Compact20Dense2LUT6Student,
        "encoder_mode": "quantile", "encoder_channels": 6,
        "conv_channels": (28, 56), "dense_width": 7040,
        "group_size": 704, "quantile_bits": 2, "lbp_bits": 0,
    },
    "tinyconv_lbp2_wide10_dense2": {
        "class": TinyConvLBP2Wide10Dense2LUT6Student,
        "encoder_mode": "lbp2", "encoder_channels": 6,
        "conv_channels": (35, 70), "dense_width": 8800,
        "group_size": 880, "quantile_bits": 0, "lbp_bits": 2,
    },
}


def _load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # Compatibility with older PyTorch.
        return torch.load(path, map_location="cpu")


def _tensor(state, key, shape, *, boolean=False, upper=None):
    value = state.get(key)
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(shape):
        raise ValueError(f"{key} must be a tensor with shape {tuple(shape)}.")
    value = value.detach().cpu()
    if boolean:
        if value.dtype != torch.bool:
            raise ValueError(f"{key} must contain Boolean truth-table values.")
        return value.clone()
    if value.dtype == torch.bool or value.is_floating_point() or value.is_complex():
        raise ValueError(f"{key} must contain integer connection indices.")
    value = value.long()
    if bool((value < 0).any()) or (upper is not None and bool((value >= upper).any())):
        raise ValueError(f"{key} contains an out-of-range connection index.")
    return value.clone()


def _validate_discrete(raw: dict) -> dict:
    if not isinstance(raw, dict) or raw.get("student_model") not in _SUPPORTED_MODELS:
        raise ValueError("Exporter requires a supported two-dense LUT6 student checkpoint.")
    spec = _SUPPORTED_MODELS[raw["student_model"]]
    encoder = raw.get("encoder")
    if not isinstance(encoder, dict) or encoder.get("mode") != spec["encoder_mode"]:
        raise ValueError("Student checkpoint has an incompatible encoder description.")
    encoder_state = encoder.get("state")
    if not isinstance(encoder_state, dict):
        raise ValueError("Student checkpoint is missing encoder state.")
    thresholds = encoder_state.get("thresholds")
    if spec["quantile_bits"]:
        if not isinstance(thresholds, torch.Tensor) or tuple(thresholds.shape) != (3, 2):
            raise ValueError("Encoder thresholds must have shape [3,2].")
    elif thresholds is not None or encoder_state:
        raise ValueError("LBP2-only encoder must not contain quantile state.")

    conv1, conv2, dense2 = raw.get("conv1"), raw.get("conv2"), raw.get("dense2")
    if not all(isinstance(item, dict) for item in (conv1, conv2, dense2)):
        raise ValueError("Student checkpoint is missing convolutional or Dense2 state.")
    conv1_channels, conv2_channels = spec["conv_channels"]
    expected_conv = (
        (conv1, spec["encoder_channels"], conv1_channels),
        (conv2, conv1_channels, conv2_channels),
    )
    normalized_convs = []
    for index, (conv, in_channels, out_channels) in enumerate(expected_conv, start=1):
        if (conv.get("in_channels"), conv.get("out_channels"), conv.get("kernel_size"),
                conv.get("stride"), conv.get("padding")) != (in_channels, out_channels, 3, 2, 1):
            raise ValueError(f"Conv{index} geometry does not match the supported student.")
        normalized_convs.append({
            **{key: conv[key] for key in (
                "in_channels", "out_channels", "kernel_size", "stride", "padding",
            )},
            "connections": _tensor(
                conv, "connections", (out_channels, 6), upper=in_channels * 9,
            ),
            "truth_tables": _tensor(conv, "truth_tables", (out_channels, 64), boolean=True),
        })

    tau = raw.get("student_tau")
    if not isinstance(tau, (int, float)) or not math.isfinite(tau) or tau <= 0:
        raise ValueError("student_tau must be finite and positive.")
    dense_width, group_size = spec["dense_width"], spec["group_size"]
    head_input_shape = (conv2_channels, 8, 8)
    if raw.get("group_size") != group_size or raw.get("groupsum_source") != "dense2":
        raise ValueError(f"Expected GroupSum(10,{group_size}) sourced from Dense2.")
    if tuple(raw.get("head_input_shape", ())) != head_input_shape:
        raise ValueError(f"Dense1 input shape must be {head_input_shape}.")
    return {
        "student_model": raw["student_model"],
        "encoder": {
            "mode": spec["encoder_mode"], "quantile_bits": spec["quantile_bits"],
            "lbp_bits": spec["lbp_bits"],
            "state": {key: value.detach().cpu().clone() for key, value in encoder_state.items()},
        },
        "conv1": normalized_convs[0],
        "conv2": normalized_convs[1],
        "connections": _tensor(
            raw, "connections", (dense_width, 6), upper=conv2_channels * 8 * 8,
        ),
        "truth_tables": _tensor(raw, "truth_tables", (dense_width, 64), boolean=True),
        "dense2": {
            "connections": _tensor(
                dense2, "connections", (dense_width, 6), upper=dense_width,
            ),
            "truth_tables": _tensor(
                dense2, "truth_tables", (dense_width, 64), boolean=True,
            ),
        },
        "flatten_order": "channel,row,column",
        "head_input_shape": head_input_shape,
        "student_tau": float(tau),
        "group_size": group_size,
        "groupsum_source": "dense2",
        "address_pin0_is_lsb": True,
    }


def _from_training_checkpoint(checkpoint: dict):
    config, student_state = checkpoint.get("config"), checkpoint.get("student_state")
    if not isinstance(config, dict) or not isinstance(student_state, dict):
        raise ValueError("Checkpoint is neither a discrete student nor a training checkpoint.")
    student_model = config.get("student_model")
    if student_model not in _SUPPORTED_MODELS:
        raise ValueError("Exporter supports only registered two-dense LUT6 students.")
    thresholds = student_state.get("encoder.thresholds")
    if _SUPPORTED_MODELS[student_model]["quantile_bits"]:
        if not isinstance(thresholds, torch.Tensor) or tuple(thresholds.shape) != (3, 2):
            raise ValueError("Training checkpoint has no valid Q2 thresholds.")
    elif thresholds is not None:
        raise ValueError("LBP2-only checkpoint unexpectedly contains Q2 thresholds.")
    model = _SUPPORTED_MODELS[student_model]["class"](
        thresholds=thresholds,
        mapping_candidates=int(config.get("mapping_candidates", 16)),
        mapping_temperature=float(config.get("mapping_temperature", 1.0)),
        student_tau=float(config.get("student_tau", 40.0)),
        node_chunk=int(config.get("node_chunk", 256)),
    )
    model.load_state_dict(student_state, strict=True)
    model.eval()
    metadata = {
        "checkpoint_kind": "best_student",
        "epoch": checkpoint.get("epoch"),
        "validation": checkpoint.get("validation"),
        "config": config,
    }
    return _validate_discrete(model.discrete_state()), metadata, model


def load_student_checkpoint(path):
    """Return validated discrete state, metadata, and an optional reference model."""
    path = Path(path).expanduser().resolve()
    checkpoint = _load(path)
    if not isinstance(checkpoint, dict):
        raise ValueError("Student checkpoint must be a dictionary.")
    if checkpoint.get("student_model") in _SUPPORTED_MODELS:
        return _validate_discrete(checkpoint), {
            "checkpoint_kind": "discrete_student", "epoch": None,
            "validation": None, "config": None,
        }, None
    return _from_training_checkpoint(checkpoint)
