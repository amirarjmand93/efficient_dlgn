"""MRLB-v2: constrained two-input relational branches on CompactLogic backends."""

import torch

from .classrouted_compact import build_model_cifar10_small_classrouted_compact_head
from .compact import CandidateLogicLayer


class RoutedSpatialCandidates(CandidateLogicLayer):
    """Candidate operands stay in per-output channel banks through resampling.

    ``channel_ranges[side, row]`` holds an inclusive start/exclusive end. The
    learned logits, Boolean opcodes, spatial sampler, and optimizer resets are
    the existing CompactLogic mechanisms; only channel selection is constrained.
    """

    def __init__(self, *, channel_ranges, **kwargs):
        super().__init__(**kwargs)
        if not self.is_convolution:
            raise ValueError("MRLB routing requires a spatial layer.")
        bounds = torch.as_tensor(channel_ranges, dtype=torch.long, device=self.weights.device)
        if bounds.shape != (2, self.out_dim, 2):
            raise ValueError("Channel ranges must have shape [2,out_channels,2].")
        if bool(((bounds[..., 0] < 0) | (bounds[..., 1] > self.in_shape[0])
                 | (bounds[..., 0] >= bounds[..., 1])).any()):
            raise ValueError("Invalid MRLB operand channel range.")
        self.register_buffer("channel_ranges", bounds.clone())
        self._sample_channels(torch.ones_like(self.weights, dtype=torch.bool))

    @torch.no_grad()
    def _sample_channels(self, slots):
        low = self.channel_ranges[..., 0, None]
        span = self.channel_ranges[..., 1, None] - low
        channels = low + (torch.rand((2, *self.weights.shape), device=self.weights.device)
                          * span).long()
        for side in range(2):
            self.impl.offsets_ch[side, slots] = channels[side, slots].to(self.impl.offsets_ch.dtype)
        # Two operands must not refer to the identical pixel. Change only a
        # replaced slot, leaving strong winners and stable rows bit-for-bit intact.
        conflict = slots & (self.impl.offsets_ch[0] == self.impl.offsets_ch[1])
        conflict &= (self.impl.offsets_h[0] == self.impl.offsets_h[1])
        conflict &= (self.impl.offsets_w[0] == self.impl.offsets_w[1])
        radius = self.impl.padding
        h = self.impl.offsets_h[1, conflict].long()
        self.impl.offsets_h[1, conflict] = ((h + radius + 1) % (2 * radius + 1) - radius).to(
            self.impl.offsets_h.dtype)

    @torch.no_grad()
    def resample(self, rows, optimizer):
        maximum, winner = self.weights.softmax(-1).max(-1)
        selected = torch.zeros_like(maximum, dtype=torch.bool)
        selected[rows] = True
        weak = selected & (maximum < 0.4)
        strong = selected & (maximum > 0.95)
        slots = (weak | strong)[:, None].expand_as(self.weights).clone()
        strong_rows = strong.nonzero(as_tuple=True)[0]
        slots[strong_rows, winner[strong_rows]] = False
        # Includes the established weak/strong LUT updates and Adam row resets.
        counts = super().resample(rows, optimizer)
        if bool(slots.any()):
            self._sample_channels(slots)
        return counts


class MRLBCompactModel(torch.nn.Sequential):
    """Parallel local/cross-depth relations; original concat bypasses stay intact."""

    concatenation_skips = (
        "Conv1+Conv2+Local+CrossDepth -> Conv3", "Conv3+Conv4 -> Dense1",
    )
    backbone_conv_indices = (1, 2, 5, 6)

    def forward(self, x):
        shallow = self[1](self[0](x))
        deep = self[2](shallow)
        original = torch.cat((shallow, deep), dim=1)
        local = self[3](deep)
        cross = self[4](original)
        conv3 = self[5](torch.cat((original, local, cross), dim=1))
        conv4 = self[6](conv3)
        features = self[7](torch.cat((conv3, conv4), dim=1))
        return self[10](self[9](self[8](features)))

    def relational_bridge_report(self):
        roles = self[5].route_ids
        return {
            "version": "MRLB-v2",
            "branch_channels": {"local": 64, "cross_depth": 64},
            "spatial_radius": {"local": 1, "cross_depth": 2},
            "conv3_input_channels": 384,
            "conv3_output_routes": {
                name: int((roles == role).sum())
                for role, name in enumerate(("original", "local_to_original", "cross_to_original"))
            },
            "operand_routing": "A uses the assigned bank; B uses original Conv1+Conv2 features",
            "channel_order": "Conv1[0:128], Conv2[128:256], Local[256:320], CrossDepth[320:384]",
            "freezing_order": ["Conv1", "Conv2", "Local", "CrossDepth", "Conv3", "Conv4"],
        }


def build_model_cifar10_small_mrlb_compact_head(
    device, *, parametrization="compact", input_thresholds=None,
    input_encoding="hybrid", num_quantile_bits=2, num_sine_bits=2,
    sine_sigma=1.0, sine_seed=0, compact_backend="auto",
    mrlb_conv3_relational_channels=128,
):
    """Add two 64-channel Boolean branches without altering the baseline builder.

    Default Conv3 budget: 384 original-only outputs, 64 local/original, and 64
    cross-depth/original. The relational rows are spread through the 512 outputs.
    Branch gate operations remain learnable: routing reserves candidate inputs,
    not a guarantee that every winning Boolean function uses both operands.
    """
    budget = mrlb_conv3_relational_channels
    if not isinstance(budget, int) or isinstance(budget, bool) or not 2 <= budget <= 510 or budget % 2:
        raise ValueError("mrlb_conv3_relational_channels must be even, from 2 to 510.")
    baseline = build_model_cifar10_small_classrouted_compact_head(
        device, parametrization=parametrization, input_thresholds=input_thresholds,
        input_encoding=input_encoding, num_quantile_bits=num_quantile_bits,
        num_sine_bits=num_sine_bits, sine_sigma=sine_sigma, sine_seed=sine_seed,
        compact_backend=compact_backend,
    )
    options = dict(device=device, backend=compact_backend)
    local = RoutedSpatialCandidates(
        **options, in_dim=128 * 16 * 16, out_dim=64, in_shape=(128, 16, 16),
        channel_ranges=torch.tensor([0, 128]).expand(2, 64, 2),
    )
    cross_bounds = torch.empty(2, 64, 2, dtype=torch.long)
    cross_bounds[0] = torch.tensor([0, 128])
    cross_bounds[1] = torch.tensor([128, 256])
    cross = RoutedSpatialCandidates(
        **options, in_dim=256 * 16 * 16, out_dim=64, in_shape=(256, 16, 16),
        kernel_size=5, channel_ranges=cross_bounds,
    )
    roles = torch.zeros(512, dtype=torch.long)
    routed_rows = torch.arange(budget) * 512 // budget
    roles[routed_rows] = torch.arange(budget) % 2 + 1
    conv3_bounds = torch.tensor([0, 256]).expand(2, 512, 2).clone()
    conv3_bounds[0, roles == 1] = torch.tensor([256, 320])
    conv3_bounds[0, roles == 2] = torch.tensor([320, 384])
    conv3 = RoutedSpatialCandidates(
        **options, in_dim=384 * 16 * 16, out_dim=512, in_shape=(384, 16, 16),
        stride=2, channel_ranges=conv3_bounds,
    )
    conv3.register_buffer("route_ids", roles.to(conv3.weights.device))
    return MRLBCompactModel(
        baseline[0], baseline[1], baseline[2], local, cross, conv3,
        baseline[4], baseline[5], baseline[6], baseline[7], baseline[8],
    ).to(device)
