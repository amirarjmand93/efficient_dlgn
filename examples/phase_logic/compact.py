"""CompactLogic candidate layers and its two-stage training controller.

The CUDA backend uses the optional, MIT-licensed upstream CompactLogic package.
The PyTorch backend is a reference implementation for CPU checks, not a claim
of CUDA performance parity. Algorithm and gate numbering follow:
https://github.com/eth-sri/CompactLogic (019677a79a1a3039e3c2c7f26689f3bcec8e22f9).
"""

import importlib
import math
from functools import lru_cache

import torch
from torch.utils.checkpoint import checkpoint


class UniformThermometerEncoder(torch.nn.Module):
    """Uniform thresholds, in the upstream threshold-major RGB channel order."""

    def __init__(self, num_bits=3):
        super().__init__()
        self.num_bits = num_bits
        self.num_quantile_bits = 0
        self.num_sine_bits = 0
        self.output_channels = 3 * num_bits
        self.channel_order = "threshold-major: R_t1 G_t1 B_t1 R_t2 G_t2 B_t2 ..."
        thresholds = torch.arange(1, num_bits + 1, dtype=torch.float32) / (num_bits + 1)
        self.register_buffer("thresholds", thresholds.expand(3, -1).clone())

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError("Thermometer encoder expects [batch,3,height,width].")
        return (x[:, None] > self.thresholds.T[None, :, :, None, None]).flatten(1, 2).to(x.dtype)


def _boolean_operation(a, b, codes):
    """Multilinear extension of upstream gate IDs 0..15 (ID 3 is identity A)."""
    t00 = ((codes >> 3) & 1).to(a.dtype)
    t01 = ((codes >> 2) & 1).to(a.dtype)
    t10 = ((codes >> 1) & 1).to(a.dtype)
    t11 = (codes & 1).to(a.dtype)
    return t00 + (t10 - t00) * a + (t01 - t00) * b + (t11 - t10 - t01 + t00) * a * b


class _TorchCandidates(torch.nn.Module):
    """Reference candidate triples with dense or one-channel spatial inputs."""

    def __init__(self, in_dim, out_dim, *, in_shape=None, stride=1, num_gates=16,
                 kernel_size=3):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.in_shape = in_shape
        self.stride = stride
        self.num_gates = num_gates
        self.padding = kernel_size // 2
        self.weights = torch.nn.Parameter(torch.zeros(out_dim, num_gates))
        with torch.no_grad():
            self.weights[:, 3] = math.log(9 * (num_gates - 1))
        self.register_buffer("gate_sequence", torch.arange(num_gates).remainder(16).to(torch.uint8).expand(out_dim, -1).clone())
        if in_shape is None:
            if 2 * out_dim < in_dim:
                raise ValueError("Dense candidate initialization requires 2*out_dim >= in_dim.")
            indices = []
            for _ in range(num_gates):
                slots = torch.randperm(2 * out_dim) % in_dim
                indices.append(torch.randperm(in_dim)[slots].reshape(2, out_dim))
            self.register_buffer("indices", torch.stack(indices, dim=-1))
        else:
            channels = torch.arange(out_dim).remainder(in_shape[0])
            self.register_buffer("offsets_ch", channels[None, :, None].expand(2, -1, num_gates).clone())
            self.register_buffer("offsets_h", torch.randint(-self.padding, self.padding + 1, (2, out_dim, num_gates)))
            self.register_buffer("offsets_w", torch.randint(-self.padding, self.padding + 1, (2, out_dim, num_gates)))
            conflict = (self.offsets_h[0] == self.offsets_h[1]) & (self.offsets_w[0] == self.offsets_w[1])
            while conflict.any():
                self.offsets_h[1, conflict] = torch.randint(-self.padding, self.padding + 1, (int(conflict.sum()),))
                self.offsets_w[1, conflict] = torch.randint(-self.padding, self.padding + 1, (int(conflict.sum()),))
                conflict = (self.offsets_h[0] == self.offsets_h[1]) & (self.offsets_w[0] == self.offsets_w[1])

    def weights_entropy(self):
        log_p = self.weights.log_softmax(-1)
        return -(log_p.exp() * log_p).sum(-1) / math.log(self.num_gates)

    def _chunk(self, x, weights, start, stop, spatial):
        codes = self.gate_sequence[start:stop]
        if spatial is None:
            index = self.indices[:, start:stop]
        else:
            height, width = self.in_shape[1:]
            pad = self.padding
            offset = (self.offsets_ch[:, start:stop] * (height + 2 * pad) * (width + 2 * pad)
                      + (self.offsets_h[:, start:stop] + pad) * (width + 2 * pad)
                      + self.offsets_w[:, start:stop] + pad)
            index = offset[..., None] + spatial
            codes = codes[..., None]
        if self.training:
            a, b = x[:, index[0]], x[:, index[1]]
            values = _boolean_operation(a, b, codes)
            probabilities = weights.softmax(-1)
            if spatial is not None:
                probabilities = probabilities[..., None]
            return (values * probabilities).sum(2)
        winners = weights.argmax(-1)
        row = torch.arange(stop - start, device=x.device)
        index = index[:, row, winners]
        codes = codes[row, winners]
        return _boolean_operation(x[:, index[0]], x[:, index[1]], codes)

    def forward(self, x):
        spatial = None
        if self.in_shape is not None:
            if tuple(x.shape[1:]) != tuple(self.in_shape):
                raise ValueError(f"Expected input shape {self.in_shape}, got {tuple(x.shape[1:])}.")
            height, width = self.in_shape[1:]
            h = torch.arange(0, height, self.stride, device=x.device)
            w = torch.arange(0, width, self.stride, device=x.device)
            pad = self.padding
            spatial = (h[:, None] * (width + 2 * pad) + w).flatten()
            x = torch.nn.functional.pad(x, (pad, pad, pad, pad)).flatten(1)
        elif x.ndim != 2 or x.shape[1] != self.in_dim:
            raise ValueError(f"Expected [batch,{self.in_dim}] dense inputs.")
        chunks = []
        chunk_size = 128 if spatial is not None else 2048
        for start in range(0, self.out_dim, chunk_size):
            stop = min(start + chunk_size, self.out_dim)
            # Bind loop variables: checkpoint recomputes this chunk in backward.
            def run(features, weights, start=start, stop=stop):
                return self._chunk(features, weights, start, stop, spatial)
            if self.training and torch.is_grad_enabled():
                output = checkpoint(run, x, self.weights[start:stop], use_reentrant=False)
            else:
                output = run(x, self.weights[start:stop])
            chunks.append(output)
        result = torch.cat(chunks, dim=1)
        if spatial is not None:
            result = result.reshape(x.shape[0], self.out_dim, len(h), len(w))
        return result

    @torch.no_grad()
    def resample_connections(self, rows, **kwargs):
        probabilities = self.weights.softmax(-1)
        maximum, winner = probabilities.max(-1)
        selected = torch.zeros_like(maximum, dtype=torch.bool)
        selected[rows] = True
        weak = selected & (maximum < kwargs.get("threshold_low", 0.4))
        strong = selected & (maximum > kwargs.get("threshold_high", 0.95))
        slots = (weak | strong)[:, None].expand_as(self.weights).clone()
        strong_rows = strong.nonzero(as_tuple=True)[0]
        slots[strong_rows, winner[strong_rows]] = False
        self.weights[weak] = torch.randn_like(self.weights[weak])
        self.weights[strong] = 0
        self.weights[strong_rows, winner[strong_rows]] = math.log(9 * (self.num_gates - 1))
        count = int(slots.sum())
        self.gate_sequence[slots] = torch.randint(16, (count,), device=self.weights.device, dtype=torch.uint8)
        if self.in_shape is None:
            for side in range(2):
                self.indices[side, slots] = torch.randint(self.in_dim, (count,), device=self.weights.device)
        else:
            # Match upstream: refresh spatial offsets while retaining channel assignment.
            for side in range(2):
                self.offsets_h[side, slots] = torch.randint(-self.padding, self.padding + 1, (count,), device=self.weights.device)
                self.offsets_w[side, slots] = torch.randint(-self.padding, self.padding + 1, (count,), device=self.weights.device)
        return weak, strong


@lru_cache(maxsize=1)
def _cuda_backend():
    try:
        return importlib.import_module("compactlogic")
    except (ImportError, OSError, RuntimeError) as error:
        raise RuntimeError(
            "The CompactLogic CUDA backend requires its optional CUDA extensions. "
            "Install using: python -m pip install --no-build-isolation "
            "'git+https://github.com/eth-sri/CompactLogic.git@019677a79a1a3039e3c2c7f26689f3bcec8e22f9' "
            "(with a matching CUDA toolkit). Use --compact-backend torch for CPU reference checks."
        ) from error


class CandidateLogicLayer(torch.nn.Module):
    """Adapter preserving freezing in state_dict and train/eval transitions."""

    def __init__(self, *, device, backend, in_dim, out_dim, in_shape=None, stride=1,
                 kernel_size=3):
        super().__init__()
        if kernel_size < 3 or kernel_size % 2 != 1:
            raise ValueError("Compact spatial kernels require an odd kernel_size >= 3.")
        device = torch.device(device)
        self.backend = "cuda" if backend == "auto" and device.type == "cuda" else backend
        if self.backend == "auto":
            self.backend = "torch"
        self.is_convolution = in_shape is not None
        self.num_gates = 16
        self.lut_rank = 2
        self.out_dim = out_dim
        self.in_shape = in_shape
        self.stride = stride
        self.resample_preferred_ops = ()
        if self.backend == "cuda":
            if device.type != "cuda":
                raise ValueError("compact-backend=cuda requires a CUDA device.")
            upstream = _cuda_backend()
            if in_shape is None:
                self.impl = upstream.LogicLayer(in_dim, out_dim, num_gates=16, residual_init=True)
            else:
                self.impl = upstream.ConvLayer(in_shape=in_shape, padding=kernel_size // 2, c_out=out_dim,
                                               ks=kernel_size, stride=stride, num_chn=1,
                                               num_gates=16, residual_init=True)
        elif self.backend == "torch":
            self.impl = _TorchCandidates(in_dim, out_dim, in_shape=in_shape, stride=stride,
                                         kernel_size=kernel_size)
        else:
            raise ValueError(f"Unknown CompactLogic backend {backend!r}.")
        self.register_buffer("frozen", torch.tensor(False))
        self._is_frozen = False
        self.register_load_state_dict_post_hook(self._restore_frozen_state)
        self.to(device)

    @property
    def weights(self):
        return self.impl.weights

    @property
    def is_frozen(self):
        return self._is_frozen

    def _restore_frozen_state(self, module, incompatible_keys):
        self._is_frozen = bool(self.frozen)
        self.weights.requires_grad_(not self.is_frozen)
        self.impl.train(self.training and not self.is_frozen)

    def freeze(self):
        self.frozen.fill_(True)
        self._is_frozen = True
        self.weights.requires_grad_(False)
        self.impl.eval()

    def forward(self, x):
        self.impl.train(self.training and not self.is_frozen)
        return self.impl(x)

    def weights_entropy(self):
        return self.impl.weights_entropy()

    @torch.no_grad()
    def resample(self, rows, optimizer):
        preferred = getattr(self, "resample_preferred_ops", ())
        efficient_dense = getattr(self, "resample_efficient_dense", False)
        # Save the winner before the backend renormalizes strong-row logits.
        winners = self.weights.argmax(-1) if preferred or efficient_dense else None
        weak, strong = self.impl.resample_connections(
            rows, threshold_high=0.95, threshold_low=0.4,
            keep_strong_weights=False,
        )
        if preferred:
            slots = (weak | strong)[:, None].expand_as(self.weights).clone()
            strong_rows = strong.nonzero(as_tuple=True)[0]
            slots[strong_rows, winners[strong_rows]] = False
            # Keep 75% of the backend's uniform samples; overwrite only 25%.
            codes = self.impl.gate_sequence[slots]
            if codes.numel():
                biased = torch.rand(codes.shape, device=codes.device) < 0.25
                pool = codes.new_tensor(preferred)
                choices = torch.randint(len(preferred), codes.shape, device=codes.device)
                self.impl.gate_sequence[slots] = torch.where(biased, pool[choices], codes)
        elif efficient_dense:
            slots = (weak | strong)[:, None].expand_as(self.weights).clone()
            strong_rows = strong.nonzero(as_tuple=True)[0]
            slots[strong_rows, winners[strong_rows]] = False
            codes = self.impl.gate_sequence[slots]
            if codes.numel():
                # Layer-specific disjoint mixture; remaining slots stay uniform.
                identity_fraction, not_fraction, constant_fraction = efficient_dense
                not_end = identity_fraction + not_fraction
                constant_end = not_end + constant_fraction
                mixture = torch.rand(codes.shape, device=codes.device)
                side = torch.randint(2, codes.shape, device=codes.device)
                identity = codes.new_tensor((3, 5))[side]
                inversion = codes.new_tensor((12, 10))[side]
                constant = codes.new_tensor((15, 0))[side]
                codes = torch.where(mixture < identity_fraction, identity, codes)
                codes = torch.where((mixture >= identity_fraction) & (mixture < not_end), inversion, codes)
                codes = torch.where((mixture >= not_end) & (mixture < constant_end), constant, codes)
                self.impl.gate_sequence[slots] = codes
        # Upstream resets whole changed rows, including their renormalized winner.
        mask = (weak | strong)[:, None].expand_as(self.weights)
        state = optimizer.state.get(self.weights, {})
        for name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            if name in state:
                state[name].masked_fill_(mask, 0)
        return int(weak.sum()), int(strong.sum())


class CompactLogicController:
    """Per-step EMA/patience resampling, followed by sequential conv freezing."""

    def __init__(self, model, config):
        self.layers = [(name, layer) for name, layer in model.named_modules()
                       if isinstance(layer, CandidateLogicLayer)]
        self.convs = [(name, layer) for name, layer in self.layers if layer.is_convolution]
        efficient = getattr(config, "compact_efficient_layer_aware_resampling", False)
        aggressive = getattr(config, "compact_aggressive_layer_aware_resampling", False)
        if aggressive and (efficient or getattr(config, "compact_layer_aware_resampling", False)):
            raise ValueError("Choose only one layer-aware resampling strategy.")
        if efficient and getattr(config, "compact_layer_aware_resampling", False):
            raise ValueError("Choose only one layer-aware resampling strategy.")
        if getattr(config, "compact_layer_aware_resampling", False):
            dense = [layer for _, layer in self.layers if not layer.is_convolution]
            if len(self.convs) != 4 or len(dense) != 2:
                raise ValueError("Layer-aware resampling requires four convolutions and two dense layers.")
            for _, layer in self.convs[:3]:
                layer.resample_preferred_ops = (1, 7)  # AND / OR
            dense[1].resample_preferred_ops = (6, 9)  # XOR / XNOR
        if efficient:
            dense = [layer for _, layer in self.layers if not layer.is_convolution]
            if len(self.convs) != 4 or len(dense) != 2:
                raise ValueError("Efficient layer-aware resampling requires four convolutions and two dense layers.")
            for _, layer in self.convs[:3]:
                layer.resample_preferred_ops = (1, 7)  # Same 25% AND / OR policy.
            dense[0].resample_efficient_dense = (0.18, 0.08, 0.04)
            dense[1].resample_efficient_dense = (0.12, 0.06, 0.02)
        if aggressive:
            dense = [layer for _, layer in self.layers if not layer.is_convolution]
            if len(self.convs) != 4 or len(dense) != 2:
                raise ValueError("Aggressive layer-aware resampling requires four convolutions and two dense layers.")
            for _, layer in self.convs[:3]:
                layer.resample_preferred_ops = (1, 7)
            dense[0].resample_efficient_dense = (0.30, 0.05, 0.15)
            dense[1].resample_efficient_dense = (0.25, 0.05, 0.10)
        self.exploration_steps = config.compact_exploration_steps
        self.resample_patience = config.compact_resample_patience
        self.freeze_patience = config.compact_freeze_patience
        self.freeze_enabled = getattr(config, "compact_freeze_mode", "entropy") != "none"
        self.ema = {}
        self.counters = {}
        self.freeze_ema = {}
        self.freeze_counters = {}
        self.weak_rows = 0
        self.strong_rows = 0

    @torch.no_grad()
    def step(self, step, optimizer):
        if step <= self.exploration_steps:
            for name, layer in self.layers:
                if layer.is_frozen:
                    continue
                entropy = layer.weights_entropy().detach()
                if name not in self.ema:
                    self.ema[name] = entropy.clone()
                    self.counters[name] = torch.zeros_like(entropy, dtype=torch.int32)
                    continue
                old = self.ema[name]
                stable = (entropy - old).abs() < 5e-4
                self.ema[name] = 0.99 * old + 0.01 * entropy
                count = torch.where(stable, self.counters[name] + 1, 0)
                rows = (count >= self.resample_patience).nonzero(as_tuple=True)[0]
                if rows.numel():
                    weak, strong = layer.resample(rows, optimizer)
                    self.weak_rows += weak
                    self.strong_rows += strong
                    count[rows] = 0
                self.counters[name] = count
        elif self.freeze_enabled:
            for name, layer in self.convs:
                if layer.is_frozen:
                    continue
                entropy = float(layer.weights_entropy().mean())
                if name not in self.freeze_ema:
                    self.freeze_ema[name] = entropy
                    self.freeze_counters[name] = 0
                    break
                old = self.freeze_ema[name]
                self.freeze_ema[name] = 0.99 * old + 0.01 * entropy
                self.freeze_counters[name] = self.freeze_counters[name] + 1 if abs(entropy - old) < 5e-4 else 0
                if self.freeze_counters[name] >= self.freeze_patience:
                    layer.freeze()
                    print(f"  compact freeze step={step} layer={name}")
                break

    def summary(self):
        return {"weak_rows_resampled": self.weak_rows,
                "strong_rows_refreshed": self.strong_rows,
                "frozen_conv_layers": [name for name, layer in self.convs if layer.is_frozen],
                "exploration_steps": self.exploration_steps}

    def state_dict(self):
        return {"ema": self.ema, "counters": self.counters,
                "freeze_ema": self.freeze_ema, "freeze_counters": self.freeze_counters,
                **self.summary()}

    def load_state_dict(self, state):
        devices = {name: layer.weights.device for name, layer in self.layers}
        for field in ("ema", "counters"):
            setattr(self, field, {name: tensor.to(devices[name])
                                 for name, tensor in state[field].items()})
        self.freeze_ema = dict(state["freeze_ema"])
        self.freeze_counters = dict(state["freeze_counters"])
        self.weak_rows = state["weak_rows_resampled"]
        self.strong_rows = state["strong_rows_refreshed"]
