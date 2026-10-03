"""Modified LogicDense: dual-path rank-2 LUT-tree dense layer.

Each output is computed by three rank-2 LUTs:

    u_j = LUT_Aj(x[a_j], x[b_j])
    v_j = LUT_Bj(x[c_j], x[d_j])
    y_j = LUT_Fj(u_j, v_j)

This gives each final output an effective receptive set of up to four input
features while keeping every gate rank-2 and reusing Torchlogix LogicDense.
"""

import torch

from .dense import LogicDense


class _PairFusionConnections(torch.nn.Module):
    """Connect output j to feature j from each concatenated branch."""

    def __init__(self, out_dim: int, device: str = "cpu"):
        super().__init__()
        out_dim = int(out_dim)
        first = torch.arange(out_dim, dtype=torch.long)
        second = first + out_dim
        indices = torch.stack((first, second), dim=0)
        self.register_buffer("indices", indices.to(device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, self.indices]


class _PairFusionLogicDense(LogicDense):
    """Rank-2 LogicDense that fuses corresponding branch outputs."""

    def _init_connections(self):
        if self.lut_rank != 2:
            raise NotImplementedError(
                "Pair-fusion LogicDense currently supports lut_rank=2 only."
            )
        if self.in_dim != 2 * self.out_dim:
            raise ValueError(
                "Pair-fusion requires in_dim == 2 * out_dim, "
                f"got in_dim={self.in_dim}, out_dim={self.out_dim}."
            )
        return _PairFusionConnections(
            out_dim=self.out_dim,
            device=self.device,
        )


class ModifiedLogicDense(torch.nn.Module):
    """Dual-path LUT-tree replacement for LogicDense.

    Each final output depends on two independently wired rank-2 branch LUTs,
    then a third rank-2 LUT fuses those two branch outputs.

    This gives effective fan-in up to four original inputs while keeping the
    physical LUT rank equal to 2.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        device: str = "cpu",
        grad_factor: float = 1.0,
        lut_rank: int = 2,
        parametrization: str = "raw",
        parametrization_kwargs: dict = None,
        connections: str = "fixed",
        connections_kwargs: dict = None,
    ):
        super().__init__()

        if lut_rank != 2:
            raise NotImplementedError(
                "ModifiedLogicDense currently supports lut_rank=2 only."
            )

        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.device = device
        self.grad_factor = float(grad_factor)
        self.lut_rank = int(lut_rank)

        common_kwargs = dict(
            device=device,
            grad_factor=grad_factor,
            lut_rank=lut_rank,
            parametrization=parametrization,
            parametrization_kwargs=parametrization_kwargs,
        )

        self.branch_a = LogicDense(
            in_dim=self.in_dim,
            out_dim=self.out_dim,
            connections=connections,
            connections_kwargs=connections_kwargs,
            **common_kwargs,
        )

        self.branch_b = LogicDense(
            in_dim=self.in_dim,
            out_dim=self.out_dim,
            connections=connections,
            connections_kwargs=connections_kwargs,
            **common_kwargs,
        )

        self.fuse = _PairFusionLogicDense(
            in_dim=2 * self.out_dim,
            out_dim=self.out_dim,
            connections="fixed",
            connections_kwargs={},
            **common_kwargs,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.branch_a(x)
        b = self.branch_b(x)
        return self.fuse(torch.cat((a, b), dim=-1))

    def set_export_mode(self, enabled: bool = True):
        self.branch_a.set_export_mode(enabled)
        self.branch_b.set_export_mode(enabled)
        self.fuse.set_export_mode(enabled)

    def get_regularization_loss(self, regularizer: str):
        return (
            self.branch_a.get_regularization_loss(regularizer)
            + self.branch_b.get_regularization_loss(regularizer)
            + self.fuse.get_regularization_loss(regularizer)
        )

    def rescale_weights(self, method):
        self.branch_a.rescale_weights(method)
        self.branch_b.rescale_weights(method)
        self.fuse.rescale_weights(method)

    def extra_repr(self):
        return (
            f"in_dim={self.in_dim}, out_dim={self.out_dim}, "
            "effective_fan_in<=4, physical_lut_rank=2"
        )
