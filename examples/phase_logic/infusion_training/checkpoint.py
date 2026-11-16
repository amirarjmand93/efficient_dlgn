"""Small checkpoint helpers for progressive aggressive infusion."""
from __future__ import annotations

from pathlib import Path

import torch


def save_infusion_checkpoint(
    path,
    *,
    step,
    model,
    optimizer,
    compact_controller,
    infusion_controller,
    config,
    metrics,
    dense_freezer=None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint_kind": "progressive_aggressive_infusion_safe_v1",
        "step": int(step),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "controller_state": compact_controller.state_dict(),
        "infusion_state": infusion_controller.state_dict(),
        "config": config.to_dict() if hasattr(config, "to_dict") else dict(config),
        "metrics": dict(metrics),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    if dense_freezer is not None:
        payload["dense_freezer_state"] = dense_freezer.state_dict()
    torch.save(payload, path)
    return path


def load_infusion_checkpoint(
    path,
    *,
    model,
    optimizer,
    compact_controller,
    infusion_controller,
    dense_freezer=None,
    map_location="cpu",
) -> dict:
    path = Path(path)
    try:
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # PyTorch versions predating weights_only.
        checkpoint = torch.load(path, map_location=map_location)
    required = {
        "step", "model_state", "optimizer_state", "controller_state", "infusion_state",
    }
    if not isinstance(checkpoint, dict) or not required.issubset(checkpoint):
        raise ValueError("Checkpoint is missing progressive infusion training state.")
    model.load_state_dict(checkpoint["model_state"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    compact_controller.load_state_dict(checkpoint["controller_state"])
    infusion_controller.load_state_dict(checkpoint["infusion_state"])
    if dense_freezer is not None:
        if "dense_freezer_state" not in checkpoint:
            raise ValueError("Checkpoint has no dense-freezer state.")
        dense_freezer.load_state_dict(checkpoint["dense_freezer_state"])
    elif "dense_freezer_state" in checkpoint:
        raise ValueError("Checkpoint expects dense freezing, but it is disabled.")
    torch.set_rng_state(checkpoint["torch_rng_state"])
    if torch.cuda.is_available() and checkpoint.get("cuda_rng_state") is not None:
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
    return checkpoint
