"""Read-only wrapper around the project's existing checkpoint loader."""

from collections.abc import Mapping
import json
from pathlib import Path

import torch
from torch import nn


def read_teacher_config(checkpoint):
    """Match the existing loader: embedded config first, sibling JSON fallback.

    Deserialize on CPU to inspect metadata and state structure, without building
    a teacher or restoring any optimizer/RNG state. No checkpoint is written.
    """
    path = Path(checkpoint).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Teacher checkpoint not found: {path}")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(saved, Mapping):
        raise ValueError("Expected a teacher checkpoint dictionary.")
    state = saved
    for key in ("model_state", "model_state_dict", "state_dict"):
        if key in saved:
            state = saved[key]
            break
    if not isinstance(state, Mapping) or not state or not all(
        isinstance(value, torch.Tensor) for value in state.values()
    ):
        raise ValueError("Missing or invalid teacher model state dictionary.")
    if isinstance(saved.get("config"), Mapping):
        configuration = dict(saved["config"])
    else:
        config_file = path.parent / "config.json"
        if not config_file.is_file():
            raise ValueError("No embedded teacher configuration or sibling config.json found.")
        configuration = json.loads(config_file.read_text())
    required = {"dataset", "model_variant", "parametrization"}
    if not isinstance(configuration, Mapping) or not required.issubset(configuration):
        raise ValueError("Teacher configuration must specify dataset, model_variant and parametrization.")
    return dict(configuration)


class FrozenTeacher(nn.Module):
    def __init__(self, checkpoint, *, device, backend="auto"):
        super().__init__()
        from ..post_training.load_checkpoint import load_checkpoint

        self.model, self.report = load_checkpoint(checkpoint, device=device, backend=backend)
        if self.report["warnings"]:
            raise ValueError("Teacher scaling could not be restored unambiguously: "
                             + "; ".join(self.report["warnings"]))
        self.model.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        # Never allow parent .train() calls to relax or train the teacher.
        super().train(False)
        return self

    @torch.no_grad()
    def forward(self, images):
        # Includes the original encoder, discrete gates and saved GroupSum tau.
        # These are pre-softmax teacher logits, not raw class counts.
        return self.model(images)
