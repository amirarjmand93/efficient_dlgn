"""Training-only Boolean wire erasure."""

import torch


class BooleanWireErasure(torch.nn.Module):
    """Replace activations with the unknown Boolean value 0.5."""

    def __init__(self, p: float = 0.02):
        super().__init__()
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be in [0, 1], got {p}.")
        self.p = float(p)
        self.enabled = False
        self.export_mode = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled or self.export_mode or self.p == 0.0:
            return x
        erase = torch.rand_like(x) < self.p
        return torch.where(erase, torch.full_like(x, 0.5), x)

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = bool(enabled) and not self.export_mode

    def set_export_mode(self, enabled: bool = True) -> None:
        self.export_mode = bool(enabled)
        if self.export_mode:
            self.enabled = False


def set_wire_erasure_enabled(model: torch.nn.Module, enabled: bool) -> None:
    """Explicitly toggle all Boolean wire-erasure modules in ``model``."""

    for module in model.modules():
        if isinstance(module, BooleanWireErasure):
            module.set_enabled(enabled)
