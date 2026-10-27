"""Class-aligned connectivity for rank-2 LogicDense layers."""

import math

import torch

from ..parametrization import LightLUTParametrization
from .dense import LogicDense


class _ClassRoutedConnections(torch.nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_classes: int,
        shared_input: bool,
        shared_input_subset: bool,
        routed_input_reuse: bool,
        seed: int,
        device: str,
    ):
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.num_classes = int(num_classes)
        self.shared_input = bool(shared_input)
        self.shared_input_subset = bool(shared_input_subset)
        self.routed_input_reuse = bool(routed_input_reuse)
        self.seed = int(seed)
        self.init_method = "class-routed"
        self.register_buffer("indices", self._build_indices().to(device))

    def _build_indices(self) -> torch.Tensor:
        if self.in_dim % self.num_classes and not self.shared_input:
            raise ValueError("in_dim must be divisible by num_classes.")
        if self.out_dim % self.num_classes:
            raise ValueError("out_dim must be divisible by num_classes.")

        generator = torch.Generator(device="cpu").manual_seed(self.seed)
        out_per_class = self.out_dim // self.num_classes
        class_indices = []

        if self.shared_input:
            if out_per_class != self.in_dim and not self.shared_input_subset:
                raise ValueError(
                    "The shared-input layer requires one output per input "
                    "for each class."
                )
            for _ in range(self.num_classes):
                if self.shared_input_subset:
                    first = torch.randint(
                        self.in_dim,
                        (out_per_class,),
                        generator=generator,
                    )
                    second = torch.randint(
                        self.in_dim,
                        (out_per_class,),
                        generator=generator,
                    )
                    collisions = first == second
                    while torch.any(collisions):
                        second[collisions] = torch.randint(
                            self.in_dim,
                            (int(collisions.sum()),),
                            generator=generator,
                        )
                        collisions = first == second
                else:
                    first = torch.randperm(self.in_dim, generator=generator)
                    second = torch.randperm(self.in_dim, generator=generator)
                    while torch.any(first == second):
                        second = torch.randperm(
                            self.in_dim,
                            generator=generator,
                        )
                class_indices.append(torch.stack((first, second), dim=0))
        else:
            in_per_class = self.in_dim // self.num_classes
            if self.routed_input_reuse and out_per_class != in_per_class:
                raise ValueError(
                    "Input-reusing routes require equal per-class input and "
                    "output widths."
                )
            if not self.routed_input_reuse and 2 * out_per_class != in_per_class:
                raise ValueError(
                    "A routed reduction requires exactly two inputs per output."
                )
            for class_index in range(self.num_classes):
                if self.routed_input_reuse:
                    first = torch.randperm(in_per_class, generator=generator)
                    second = torch.randperm(in_per_class, generator=generator)
                    while torch.any(first == second):
                        second = torch.randperm(
                            in_per_class,
                            generator=generator,
                        )
                    indices = torch.stack((first, second), dim=0)
                    indices += class_index * in_per_class
                    class_indices.append(indices)
                else:
                    permutation = torch.randperm(
                        in_per_class,
                        generator=generator,
                    )
                    permutation += class_index * in_per_class
                    class_indices.append(
                        permutation.reshape(out_per_class, 2).transpose(0, 1)
                    )

        return torch.cat(class_indices, dim=1).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, self.indices]


class ClassRoutedLogicDense(LogicDense):
    """LogicDense whose fixed connections follow contiguous class routes."""

    def __init__(
        self,
        *args,
        num_classes: int = 10,
        shared_input: bool = False,
        shared_input_subset: bool = False,
        routed_input_reuse: bool = False,
        topology_seed: int = 0,
        **kwargs,
    ):
        self.num_classes = int(num_classes)
        self.shared_input = bool(shared_input)
        self.shared_input_subset = bool(shared_input_subset)
        self.routed_input_reuse = bool(routed_input_reuse)
        self.topology_seed = int(topology_seed)
        kwargs["connections"] = "fixed"
        super().__init__(*args, **kwargs)

    def _init_connections(self):
        if self.lut_rank != 2:
            raise NotImplementedError(
                "ClassRoutedLogicDense currently supports lut_rank=2 only."
            )
        return _ClassRoutedConnections(
            in_dim=self.in_dim,
            out_dim=self.out_dim,
            num_classes=self.num_classes,
            shared_input=self.shared_input,
            shared_input_subset=self.shared_input_subset,
            routed_input_reuse=self.routed_input_reuse,
            seed=self.topology_seed,
            device=self.device,
        )

    @torch.no_grad()
    def row_entropy(self) -> torch.Tensor:
        """Mean normalized LightLUT truth-table entropy for each output row."""
        if not isinstance(self.parametrization, LightLUTParametrization):
            raise ValueError("Entropy monitoring requires LightLUT parametrization.")
        probabilities = torch.sigmoid(
            self.weight.detach() / self.parametrization.temperature
        ).clamp(1e-7, 1 - 1e-7)
        entropy = -probabilities * probabilities.log()
        entropy -= (1 - probabilities) * torch.log1p(-probabilities)
        return entropy.mean(dim=1) / math.log(2)

    @torch.no_grad()
    def candidate_pairs(
        self,
        inputs: torch.Tensor,
        output_gradient: torch.Tensor,
        row: int,
        *,
        feature_pool_size: int,
        num_candidates: int,
    ) -> torch.Tensor:
        """Screen all shared features, then pair high-signal diverse features.

        The score is the minibatch covariance of one input feature with the
        selected gate's output gradient. Only this gate's score vector is
        formed, never an in_dim-by-out_dim routing matrix.
        """
        if not self.shared_input:
            raise ValueError("Candidate search requires shared-input routing.")
        if inputs.ndim != 2 or inputs.shape[1] != self.in_dim:
            raise ValueError("inputs must have shape (batch, in_dim).")
        if output_gradient.shape != (inputs.shape[0],):
            raise ValueError("output_gradient must have shape (batch,).")
        centered_gradient = output_gradient - output_gradient.mean()
        scores = torch.mv(inputs.T, centered_gradient).abs()
        pool = scores.topk(min(self.in_dim, feature_pool_size)).indices
        count = min(num_candidates, pool.numel() - 1)
        centered = inputs[:, pool] - inputs[:, pool].mean(dim=0, keepdim=True)
        norms = torch.linalg.vector_norm(centered, dim=0).clamp_min(1e-8)
        correlations = (
            centered[:, :count].T @ centered
        ).abs() / (norms[:count, None] * norms[None, :])
        partner_scores = scores[pool][None, :] * (1 - correlations)
        partner_scores.scatter_(
            1, torch.arange(count, device=pool.device)[:, None], float("-inf")
        )
        current = self.connections.indices[:, row]
        partner_scores.masked_fill_(
            ((pool[:count, None] == current[0]) & (pool[None, :] == current[1]))
            | ((pool[:count, None] == current[1]) & (pool[None, :] == current[0])),
            float("-inf"),
        )
        partner = partner_scores.argmax(dim=1)
        pairs = torch.stack((pool[:count], pool[partner]), dim=1)
        valid = torch.isfinite(partner_scores.gather(1, partner[:, None])[:, 0])
        return pairs[valid]

    @torch.no_grad()
    def replace_row(
        self,
        row: int,
        pair: torch.Tensor,
        new_weight: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        """Install a probed pair and reset only that gate's weights/moments."""
        if not self.shared_input:
            raise ValueError("Rewiring requires shared-input routing.")
        if not 0 <= row < self.out_dim:
            raise IndexError(row)
        if (
            pair.shape != (2,)
            or bool((pair < 0).any())
            or bool((pair >= self.in_dim).any())
            or bool(pair[0] == pair[1])
        ):
            raise ValueError("pair must contain two distinct, valid input indices.")
        if new_weight.shape != self.weight[row].shape:
            raise ValueError("new_weight shape does not match the LUT row.")
        self.connections.indices[:, row].copy_(pair)
        self.weight[row].copy_(new_weight)
        state = optimizer.state.get(self.weight, {})
        for name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            moment = state.get(name)
            if moment is not None:
                moment[row].zero_()
