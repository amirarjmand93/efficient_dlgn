"""Safe-point rollback and portable epoch-boundary resume for KD infusion."""

import random
from pathlib import Path

import numpy as np
import torch


def _save(state, path):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def _rng_state():
    numpy = np.random.get_state()
    return dict(torch=torch.get_rng_state(), python=random.getstate(),
                numpy=(numpy[0], numpy[1].tolist(), *numpy[2:]),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def _restore_rng(state):
    torch.set_rng_state(state["torch"])
    random.setstate(state["python"])
    numpy = state["numpy"]
    np.random.set_state((numpy[0], np.asarray(numpy[1], dtype=np.uint32), *numpy[2:]))
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _training_state(controller):
    return dict(student_state=controller.model.state_dict(),
                optimizer_state=controller.optimizer.state_dict(),
                infusion_state=controller.state_dict(), rng=_rng_state(),
                requires_grad={name: p.requires_grad for name, p in controller.model.named_parameters()})


def _restore(controller, state):
    controller.model.load_state_dict(state["student_state"], strict=True)
    controller.optimizer.load_state_dict(state["optimizer_state"])
    controller.load_state_dict(state["infusion_state"])
    for name, parameter in controller.model.named_parameters():
        parameter.requires_grad_(state["requires_grad"][name])
    _restore_rng(state["rng"])


def save_safe_checkpoint(controller):
    controller.safe_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    _save(dict(checkpoint_kind="kd_infusion_safe_v1", **_training_state(controller)),
          controller.safe_checkpoint)


def restore_safe_checkpoint(controller):
    state = torch.load(controller.safe_checkpoint, map_location="cpu", weights_only=True)
    if state.get("checkpoint_kind") != "kd_infusion_safe_v1":
        raise ValueError("Invalid KD infusion safe checkpoint.")
    _restore(controller, state)


def save_training_checkpoint(path, *, controller, config, epoch, step, best_path, loaders):
    safe = (torch.load(controller.safe_checkpoint, map_location="cpu", weights_only=True)
            if controller.safe_checkpoint.is_file() else None)
    _save(dict(checkpoint_kind="kd_infusion_training_v1", config=config,
                    epoch=epoch, step=step, **_training_state(controller), safe_checkpoint=safe,
                    best_checkpoint=torch.load(best_path, map_location="cpu", weights_only=True),
                    loader_rng=[loader.generator.get_state() for loader in loaders]), path)


def load_training_checkpoint(path, *, controller, config, best_path, loaders):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("checkpoint_kind") != "kd_infusion_training_v1":
        raise ValueError("Expected kd_infusion_latest.pt, not an inference checkpoint.")
    mutable = {"epochs", "run_name", "output_dir", "device", "num_workers", "teacher_backend",
               "evaluate_test", "kd_infusion_resume"}
    for name, value in config.items():
        if name not in mutable and state["config"].get(name) != value:
            raise ValueError(f"Resume setting differs from the checkpoint: {name}.")
    if config["epochs"] <= state["epoch"]:
        raise ValueError("Resume requires --epochs greater than the saved epoch.")
    if len(loaders) != len(state["loader_rng"]):
        raise ValueError("Resume data-loader count differs from the checkpoint.")
    _restore(controller, state)
    if state["safe_checkpoint"] is not None:
        _save(state["safe_checkpoint"], controller.safe_checkpoint)
    _save(state["best_checkpoint"], best_path)
    for loader, rng in zip(loaders, state["loader_rng"]):
        loader.generator.set_state(rng)
    return state
