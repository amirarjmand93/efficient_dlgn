"""Class-aligned fixed connectivity for rank-2 LogicDense layers."""

import torch

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
