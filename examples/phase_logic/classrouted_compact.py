"""CompactLogic CIFAR-10 heads with skip-connected and plain backbones."""

import math

import torch

from torchlogix.layers import ClassRoutedLogicDense, GroupSum

from .compact import CandidateLogicLayer


class ClassRoutedCandidateDense(torch.nn.Module):
    """Sixteen gate/input candidates per output within fixed class boundaries."""

    def __init__(self, original, *, device, backend="auto"):
        super().__init__()
        if not isinstance(original, ClassRoutedLogicDense) or original.lut_rank != 2:
            raise ValueError("Expected a rank-2 ClassRoutedLogicDense layer.")
        self.in_dim = original.in_dim
        self.out_dim = original.out_dim
        self.num_classes = original.num_classes
        self.shared_input = original.shared_input
        self.lut_rank = 2
        self.num_gates = 16
        if self.out_dim % self.num_classes or (not self.shared_input and self.in_dim % self.num_classes):
            raise ValueError("Routed widths must divide evenly into classes.")
        self.out_per_class = self.out_dim // self.num_classes
        self.in_per_class = self.in_dim if self.shared_input else self.in_dim // self.num_classes
        # The CompactLogic backbone has 32,768 shared features. A single class's
        # 4,608 outputs cannot satisfy upstream's per-engine coverage initializer
        # (2*out_dim >= in_dim). Pool the shared-input output rows into one engine;
        # all classes still select from the full input domain, and their output
        # blocks remain contiguous. Later, restricted routes remain independent.
        self.pooled_shared_engine = self.shared_input and 2 * self.out_per_class < self.in_dim
        engine_out_dim = self.out_dim if self.pooled_shared_engine else self.out_per_class
        num_engines = 1 if self.pooled_shared_engine else self.num_classes
        self.routes = torch.nn.ModuleList()
        for class_index in range(num_engines):
            route = CandidateLogicLayer(
                device=device, backend=backend,
                in_dim=self.in_per_class, out_dim=engine_out_dim,
            )
            start = class_index * engine_out_dim
            offset = 0 if self.shared_input else class_index * self.in_per_class
            # Keep the routing template's pair for the residual/identity winner.
            # Other candidates start with the backend's coverage-aware sampling.
            with torch.no_grad():
                initial = original.connections.indices[:, start:start + engine_out_dim] - offset
                if bool((initial < 0).any()) or bool((initial >= self.in_per_class).any()):
                    raise ValueError("The original connections escape their class route.")
                route.impl.indices[:, :, 3].copy_(initial.to(route.impl.indices.device))
                if route.backend == "cuda":
                    # CUDA backward needs an inverse index for the installed pairs.
                    route.impl.refresh_backward_index()
            self.routes.append(route)
        self.backend = self.routes[0].backend

    def forward(self, x):
        if x.ndim != 2 or x.shape[1] != self.in_dim:
            raise ValueError(f"Expected [batch,{self.in_dim}] routed dense inputs.")
        outputs = []
        for class_index, route in enumerate(self.routes):
            if self.shared_input:
                features = x
            else:
                start = class_index * self.in_per_class
                features = x[:, start:start + self.in_per_class]
            outputs.append(route(features))
        return torch.cat(outputs, dim=1)


class BitConcatCompactModel(torch.nn.Sequential):
    """Preserve same-resolution features without addition or detached skips."""

    concatenation_skips = ("Conv1+Conv2 -> Conv3", "Conv3+Conv4 -> Dense1")

    def forward(self, x):
        encoded = self[0](x)
        shallow16 = self[1](encoded)
        deep16 = self[2](shallow16)
        shallow8 = self[3](torch.cat((shallow16, deep16), dim=1))
        deep8 = self[4](shallow8)
        features = self[5](torch.cat((shallow8, deep8), dim=1))
        return self[8](self[7](self[6](features)))


def build_model_cifar10_small_classrouted_compact_head(
    device,
    *,
    parametrization="compact",
    input_thresholds=None,
    input_encoding="hybrid",
    num_quantile_bits=2,
    num_sine_bits=2,
    sine_sigma=1.0,
    sine_seed=0,
    compact_backend="auto",
):
    """Two same-resolution concatenation skips and shared 88k -> 88k layers."""

    # Import only when building, avoiding a model-registry import cycle.
    from .model import _build_cifar10_encoder

    if parametrization != "compact":
        raise ValueError("The candidate head requires --parametrization compact.")
    encoder = _build_cifar10_encoder(
        input_thresholds,
        input_encoding=input_encoding, num_quantile_bits=num_quantile_bits,
        num_sine_bits=num_sine_bits, sine_sigma=sine_sigma, sine_seed=sine_seed,
    )
    shape = (encoder.output_channels, 32, 32)
    layers = [encoder]
    for conv_index, (channels, stride) in enumerate(((128, 2), (128, 1), (512, 2), (512, 1))):
        if conv_index == 2:
            # Conv3 receives both Conv1 and Conv2 at 16x16 resolution.
            shape = (2 * shape[0], *shape[1:])
        layers.append(CandidateLogicLayer(
            device=device, backend=compact_backend, in_dim=math.prod(shape),
            out_dim=channels, in_shape=shape, stride=stride,
        ))
        shape = (channels, (shape[1] + stride - 1) // stride,
                 (shape[2] + stride - 1) // stride)
    # Concatenate Conv3 and Conv4 at 8x8 before the fully shared head.
    shape = (2 * shape[0], *shape[1:])
    layers.append(torch.nn.Flatten())
    layers.append(CandidateLogicLayer(
        device=device, backend=compact_backend,
        in_dim=math.prod(shape), out_dim=88_000,
    ))
    layers.append(CandidateLogicLayer(
        device=device, backend=compact_backend, in_dim=88_000, out_dim=88_000,
    ))
    return BitConcatCompactModel(*layers, GroupSum(k=10, tau=55)).to(device)


def build_model_cifar10_compact_sixconv(
    device, *, parametrization="compact", input_thresholds=None,
    input_encoding="hybrid", num_quantile_bits=2, num_sine_bits=2,
    sine_sigma=1.0, sine_seed=0, compact_backend="auto",
):
    """Plain 128/128/256/256/256/256 conv stack; no skips or concatenation."""
    from .model import _build_cifar10_encoder

    if parametrization != "compact":
        raise ValueError("The six-conv model requires --parametrization compact.")
    encoder = _build_cifar10_encoder(
        input_thresholds, input_encoding=input_encoding,
        num_quantile_bits=num_quantile_bits, num_sine_bits=num_sine_bits,
        sine_sigma=sine_sigma, sine_seed=sine_seed,
    )
    shape = (encoder.output_channels, 32, 32)
    layers = [encoder]
    for channels, stride in (
        (128, 2), (128, 1), (256, 2), (256, 1), (256, 1), (256, 1)
    ):
        layers.append(CandidateLogicLayer(
            device=device, backend=compact_backend, in_dim=math.prod(shape),
            out_dim=channels, in_shape=shape, stride=stride,
        ))
        shape = (channels, (shape[1] + stride - 1) // stride,
                 (shape[2] + stride - 1) // stride)
    layers.append(torch.nn.Flatten())
    in_dim = math.prod(shape)
    for out_dim in (80_000, 80_000):
        layers.append(CandidateLogicLayer(
            device=device, backend=compact_backend, in_dim=in_dim, out_dim=out_dim,
        ))
        in_dim = out_dim
    return torch.nn.Sequential(*layers, GroupSum(k=10, tau=40)).to(device)
