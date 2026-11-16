"""Rollback snapshots for the independent CAI training entry point."""

import random

import numpy as np
import torch


def save_safe_checkpoint(path, *, step, model, optimizer, compact_controller,
                         cai_controller, config, metrics):
    torch.save({
        "checkpoint_kind": "continuous_adaptive_infusion_safe_v1",
        "step": step,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "compact_controller_state": compact_controller.state_dict(),
        "cai_controller_state": cai_controller.state_dict(),
        "config": config,
        "metrics": metrics,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
    }, path)


def load_safe_checkpoint(path, *, model, optimizer, compact_controller, cai_controller,
                         device):
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("checkpoint_kind") != "continuous_adaptive_infusion_safe_v1":
        raise ValueError("Checkpoint is not a CAI safe snapshot.")
    model.load_state_dict(payload["model_state"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state"])
    compact_controller.load_state_dict(payload["compact_controller_state"])
    cai_controller.load_state_dict(payload["cai_controller_state"])
    torch.set_rng_state(payload["torch_rng_state"].cpu())
    if torch.cuda.is_available() and payload["cuda_rng_state"] is not None:
        torch.cuda.set_rng_state_all([state.cpu() for state in payload["cuda_rng_state"]])
    np.random.set_state(payload["numpy_rng_state"])
    random.setstate(payload["python_rng_state"])
    return payload
