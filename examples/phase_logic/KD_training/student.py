"""DWN-inspired LUT6 student; bounded mapping and multilinear/STE gradients.

This is not DWN's EFD CUDA operator or its full learnable mapping matrix.
Training and evaluation both have a hard Boolean forward pass.
"""

import torch
from torch import nn
from torch.nn import functional as F


STUDENT_MODELS = (
    "lut6",
    "tinyconv_q2_lbp2",
    "tinyconv_q2_lbp2_wide10",
    "tinyconv_q2_lbp2_dense2", #best
    "tinyconv_q2_lbp2_dense2_wide10",
    "tinyconv_q2_lbp2_dense2_6000", # worse
    "tinyconv_q2_1conv_dense2", # worse
    "tinyconv_q2_wide10_dense2",
    "tinyconv_q2_wide10_dense2_4conv",
    "tinyconv_q2_compact20_dense2",
    "tinyconv_lbp2_wide10_dense2",
)


class _MapInputs(torch.autograd.Function):
    """Hard input selection, with a softmax-mixture surrogate for mapping logits."""

    @staticmethod
    def forward(ctx, inputs, candidates, logits, temperature):
        selected = candidates.gather(-1, logits.argmax(-1, keepdim=True)).squeeze(-1)
        ctx.save_for_backward(inputs, candidates, logits)
        ctx.temperature = temperature
        return inputs[:, selected]

    @staticmethod
    def backward(ctx, grad):
        inputs, candidates, logits = ctx.saved_tensors
        probabilities = (logits / ctx.temperature).softmax(-1)
        if inputs.shape[1] <= 512:
            # Local convolution patches: avoid [batch*positions,kernels,6,K]
            # candidate expansions by accumulating correlations via matmul.
            correlations = grad.flatten(1).T @ inputs
            scores = correlations.gather(1, candidates.flatten(0, 1)).reshape_as(logits)
        else:
            # Dense head: only one node chunk is expanded during backward.
            values = inputs[:, candidates]
            scores = (values * grad.unsqueeze(-1)).sum(0)
        derivative = probabilities * (scores - (scores * probabilities).sum(-1, keepdim=True))
        input_grad = None
        if ctx.needs_input_grad[0]:
            # Baseline encoded pixels need no gradient. The new convolutional
            # features do: propagate the same soft-mixture surrogate upstream.
            if inputs.shape[1] <= 512:
                weights = inputs.new_zeros((logits.shape[0] * logits.shape[1], inputs.shape[1]))
                weights.scatter_add_(1, candidates.flatten(0, 1), probabilities.flatten(0, 1))
                input_grad = grad.flatten(1) @ weights
            else:
                input_grad = torch.zeros_like(inputs)
                sources = candidates.reshape(1, -1).expand(inputs.shape[0], -1)
                contributions = (grad.unsqueeze(-1) * probabilities).reshape(inputs.shape[0], -1)
                input_grad.scatter_add_(1, sources, contributions)
        return input_grad, None, derivative / ctx.temperature, None


class _LUTLookup(torch.autograd.Function):
    """Hard-address lookup; exact multilinear derivatives at binary vertices.

    Address convention: pin 0 is the least significant address bit. Only the
    selected table entry gets a table gradient. Pin gradients use the difference
    between neighboring real-valued table entries with that bit set/unset.
    """

    @staticmethod
    def forward(ctx, bits, tables):
        powers = torch.tensor((1, 2, 4, 8, 16, 32), device=bits.device)
        address = (bits.long() * powers).sum(-1)
        ctx.save_for_backward(address, tables, powers)
        return tables.gather(1, address.T).T

    @staticmethod
    def backward(ctx, grad):
        address, tables, powers = ctx.saved_tensors
        table_grad = torch.zeros_like(tables)
        table_grad.scatter_add_(1, address.T, grad.T)
        pin_grad = []
        for bit in powers:
            high = tables.gather(1, (address | bit).T).T
            low = tables.gather(1, (address & ~bit).T).T
            pin_grad.append((high - low) * grad)
        return torch.stack(pin_grad, -1), table_grad


def _candidate_pool(nodes, input_bits, count):
    """Disjoint per-pin candidate pools; any six selected inputs are distinct."""
    if not 1 <= count <= input_bits // 6:
        raise ValueError("Six candidate pools must fit the available inputs.")
    pool = torch.randint(input_bits, (nodes, 6 * count))
    while True:
        ordered, positions = pool.sort(-1)
        duplicate = torch.zeros_like(pool, dtype=torch.bool)
        duplicate[:, 1:] = ordered[:, 1:] == ordered[:, :-1]
        mask = torch.zeros_like(duplicate).scatter(1, positions, duplicate)
        if not mask.any():
            break
        pool[mask] = torch.randint(input_bits, (int(mask.sum()),))
    return pool.reshape(nodes, 6, count)


def _lut6_nodes(inputs, candidates, mapping_logits, tables, *, temperature, chunk, training):
    """Shared hard-forward LUT evaluation for dense nodes and spatial kernels."""
    chunks = []
    for start in range(0, tables.shape[0], chunk):
        end = min(start + chunk, tables.shape[0])
        pool, logits = candidates[start:end], mapping_logits[start:end]
        if training:
            bits = _MapInputs.apply(inputs, pool, logits, temperature)
            values = _LUTLookup.apply(bits, tables[start:end])
            hard = (values >= 0).to(values.dtype)
            result = hard + (values - values.detach())
        else:
            selected = pool.gather(-1, logits.argmax(-1, keepdim=True)).squeeze(-1)
            bits = inputs[:, selected].long()
            powers = bits.new_tensor((1, 2, 4, 8, 16, 32))
            address = (bits * powers).sum(-1)
            result = (tables[start:end].gather(1, address.T).T >= 0).float()
        chunks.append(result)
    return torch.cat(chunks, dim=1)


class LUT6Student(nn.Module):
    """RGB -> 10 thresholds/channel -> 8,000 LUT6 -> GroupSum(10)."""

    nodes = 8000
    pins = 6
    classes = 10
    input_bits = 3 * 32 * 32 * 10

    def __init__(self, *, mapping_candidates=16, mapping_temperature=1.0,
                 student_tau=40.0, node_chunk=256):
        super().__init__()
        if not 1 <= mapping_candidates <= self.input_bits // self.pins:
            raise ValueError("Invalid candidate count.")
        if mapping_temperature <= 0 or student_tau <= 0 or node_chunk < 1:
            raise ValueError("Temperatures and chunk size must be positive.")
        self.mapping_temperature = mapping_temperature
        self.student_tau = student_tau
        self.node_chunk = node_chunk
        self.register_buffer("thresholds", torch.arange(1, 11, dtype=torch.float32) / 11)
        self.register_buffer("input_candidates", _candidate_pool(self.nodes, self.input_bits, mapping_candidates))
        self.mapping_logits = nn.Parameter(torch.randn(self.nodes, self.pins, mapping_candidates) * 0.01)
        # Signed table entries; their signs are the deployed truth table bits.
        self.tables = nn.Parameter(torch.rand(self.nodes, 64) * 2 - 1)

    def encode(self, images):
        if images.ndim != 4 or tuple(images.shape[1:]) != (3, 32, 32):
            raise ValueError("Expected RGB CIFAR images [batch,3,32,32] in [0,1].")
        # Fixed uniform thermometer, channel/pixel/threshold order.
        return (images.unsqueeze(-1) > self.thresholds).flatten(1).float()

    def forward(self, images):
        inputs = self.encode(images)
        nodes = _lut6_nodes(inputs, self.input_candidates, self.mapping_logits, self.tables,
                           temperature=self.mapping_temperature, chunk=self.node_chunk, training=self.training)
        # Exactly 800 consecutive Boolean outputs per class, no learned head.
        return nodes.reshape(-1, self.classes, 800).sum(-1) / self.student_tau

    @torch.no_grad()
    def clamp_tables(self):
        self.tables.clamp_(-1, 1)

    @torch.no_grad()
    def discrete_state(self):
        """Deployable Boolean tables and fixed six-input connections (no teacher)."""
        connections = self.input_candidates.gather(
            -1, self.mapping_logits.argmax(-1, keepdim=True)
        ).squeeze(-1)
        return {"connections": connections.cpu(), "truth_tables": (self.tables >= 0).cpu(),
                "thresholds": self.thresholds.cpu().clone(), "student_tau": self.student_tau,
                "group_size": 800, "address_pin0_is_lsb": True}


class _LUT6Conv2d(nn.Module):
    """3x3, stride-2, zero-padded Boolean convolution; one shared LUT6/kernel."""

    def __init__(self, in_channels, out_channels, *, mapping_candidates, mapping_temperature, node_chunk):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.mapping_temperature, self.node_chunk = mapping_temperature, node_chunk
        self.register_buffer("input_candidates", _candidate_pool(out_channels, in_channels * 9, mapping_candidates))
        self.mapping_logits = nn.Parameter(torch.randn(out_channels, 6, mapping_candidates) * 0.01)
        self.tables = nn.Parameter(torch.rand(out_channels, 64) * 2 - 1)

    def forward(self, features):
        batch, _, height, width = features.shape
        patches = F.unfold(features, kernel_size=3, padding=1, stride=2)
        locations = patches.shape[-1]
        patches = patches.transpose(1, 2).reshape(batch * locations, -1)
        outputs = _lut6_nodes(patches, self.input_candidates, self.mapping_logits, self.tables,
                             temperature=self.mapping_temperature, chunk=self.node_chunk, training=self.training)
        return outputs.reshape(batch, locations, self.out_channels).transpose(1, 2).reshape(
            batch, self.out_channels, (height + 1) // 2, (width + 1) // 2)

    @torch.no_grad()
    def discrete_state(self):
        selected = self.input_candidates.gather(-1, self.mapping_logits.argmax(-1, keepdim=True)).squeeze(-1)
        return {"connections": selected.cpu(), "truth_tables": (self.tables >= 0).cpu(),
                "in_channels": self.in_channels, "out_channels": self.out_channels,
                "kernel_size": 3, "stride": 2, "padding": 1, "padding_value": 0,
                "patch_order": "channel,row,column", "address_pin0_is_lsb": True}


class _LBP2Encoder(nn.Module):
    """Two fixed radius-1 Boolean comparisons per RGB channel, CHW order."""

    output_channels = 6

    def forward(self, images):
        padded = F.pad(images, (1, 1, 1, 1), mode="replicate")
        horizontal = padded[:, :, 1:-1, 2:] >= padded[:, :, 1:-1, :-2]
        vertical = padded[:, :, 2:, 1:-1] >= padded[:, :, :-2, 1:-1]
        return torch.stack((horizontal, vertical), dim=2).reshape(
            images.shape[0], self.output_channels, images.shape[2], images.shape[3]
        ).to(images.dtype)


class TinyConvLUT6Student(LUT6Student):
    """Existing Q2+LBP2 -> LUT6 conv32/conv64 -> shared 8,000-LUT6 head."""

    input_bits = 64 * 8 * 8

    def __init__(self, *, thresholds, mapping_candidates=16, mapping_temperature=1.0,
                 student_tau=40.0, node_chunk=256):
        if not 1 <= mapping_candidates <= (12 * 9) // 6:
            raise ValueError("tinyconv_q2_lbp2 supports at most 18 candidates/pin in Conv1.")
        super().__init__(mapping_candidates=mapping_candidates, mapping_temperature=mapping_temperature,
                         student_tau=student_tau, node_chunk=node_chunk)
        from ..model import HybridFeatureEncoder

        del self.thresholds  # This variant uses the encoder's quantile buffers.
        self.encoder = HybridFeatureEncoder(mode="hybrid_lbp", thresholds=thresholds,
                                            num_quantile_bits=2, num_sine_bits=0)
        options = dict(mapping_candidates=mapping_candidates, mapping_temperature=mapping_temperature,
                       node_chunk=node_chunk)
        self.conv1 = _LUT6Conv2d(self.encoder.output_channels, 32, **options)
        self.conv2 = _LUT6Conv2d(32, 64, **options)

    def encode(self, images):
        if images.ndim != 4 or tuple(images.shape[1:]) != (3, 32, 32):
            raise ValueError("Expected RGB CIFAR images [batch,3,32,32] in [0,1].")
        return self.conv2(self.conv1(self.encoder(images))).flatten(1)

    @torch.no_grad()
    def clamp_tables(self):
        super().clamp_tables()
        self.conv1.tables.clamp_(-1, 1)
        self.conv2.tables.clamp_(-1, 1)

    @torch.no_grad()
    def discrete_state(self):
        selected = self.input_candidates.gather(-1, self.mapping_logits.argmax(-1, keepdim=True)).squeeze(-1)
        return {"student_model": "tinyconv_q2_lbp2",
                "encoder": {"mode": "hybrid_lbp", "quantile_bits": 2, "lbp_bits": 2,
                            "state": {key: value.cpu().clone() for key, value in self.encoder.state_dict().items()}},
                "conv1": self.conv1.discrete_state(), "conv2": self.conv2.discrete_state(),
                "connections": selected.cpu(), "truth_tables": (self.tables >= 0).cpu(),
                "flatten_order": "channel,row,column", "head_input_shape": (64, 8, 8),
                "student_tau": self.student_tau, "group_size": 800, "address_pin0_is_lsb": True}


class TinyConvWide10LUT6Student(TinyConvLUT6Student):
    """Q2+LBP2 -> LUT6 conv35/conv70 -> 8,800-LUT6 head."""

    nodes = 8800
    input_bits = 70 * 8 * 8
    group_size = 880

    def __init__(self, *, thresholds, mapping_candidates=16, mapping_temperature=1.0,
                 student_tau=40.0, node_chunk=256):
        if not 1 <= mapping_candidates <= (12 * 9) // 6:
            raise ValueError("tinyconv_q2_lbp2_wide10 supports at most 18 candidates/pin in Conv1.")
        # Initialize the widened dense head directly; TinyConvLUT6Student's
        # constructor fixes the convolution widths at 32/64.
        LUT6Student.__init__(
            self, mapping_candidates=mapping_candidates,
            mapping_temperature=mapping_temperature,
            student_tau=student_tau, node_chunk=node_chunk,
        )
        from ..model import HybridFeatureEncoder

        del self.thresholds
        self.encoder = HybridFeatureEncoder(
            mode="hybrid_lbp", thresholds=thresholds,
            num_quantile_bits=2, num_sine_bits=0,
        )
        options = dict(
            mapping_candidates=mapping_candidates,
            mapping_temperature=mapping_temperature,
            node_chunk=node_chunk,
        )
        self.conv1 = _LUT6Conv2d(self.encoder.output_channels, 35, **options)
        self.conv2 = _LUT6Conv2d(35, 70, **options)

    def forward(self, images):
        nodes = _lut6_nodes(
            self.encode(images), self.input_candidates, self.mapping_logits, self.tables,
            temperature=self.mapping_temperature, chunk=self.node_chunk, training=self.training,
        )
        return nodes.reshape(-1, self.classes, self.group_size).sum(-1) / self.student_tau

    @torch.no_grad()
    def discrete_state(self):
        state = super().discrete_state()
        state.update(
            student_model="tinyconv_q2_lbp2_wide10",
            head_input_shape=(70, 8, 8),
            group_size=self.group_size,
        )
        return state


class TinyConv2DenseLUT6Student(TinyConvLUT6Student):
    """Q2+LBP2 -> conv32/64 -> 8,000 LUT6 -> 8,000 LUT6 -> GroupSum."""

    group_size = 800
    model_name = "tinyconv_q2_lbp2_dense2"

    def __init__(self, *, thresholds, mapping_candidates=16, mapping_temperature=1.0,
                 student_tau=40.0, node_chunk=256):
        super().__init__(thresholds=thresholds, mapping_candidates=mapping_candidates,
                         mapping_temperature=mapping_temperature, student_tau=student_tau,
                         node_chunk=node_chunk)
        self.register_buffer("dense2_input_candidates", _candidate_pool(self.nodes, self.nodes, mapping_candidates))
        self.dense2_mapping_logits = nn.Parameter(torch.randn(self.nodes, self.pins, mapping_candidates) * 0.01)
        self.dense2_tables = nn.Parameter(torch.rand(self.nodes, 64) * 2 - 1)

    def forward(self, images):
        first = _lut6_nodes(self.encode(images), self.input_candidates, self.mapping_logits, self.tables,
                           temperature=self.mapping_temperature, chunk=self.node_chunk, training=self.training)
        second = _lut6_nodes(first, self.dense2_input_candidates, self.dense2_mapping_logits, self.dense2_tables,
                            temperature=self.mapping_temperature, chunk=self.node_chunk, training=self.training)
        return second.reshape(-1, self.classes, self.group_size).sum(-1) / self.student_tau

    @torch.no_grad()
    def clamp_tables(self):
        super().clamp_tables()
        self.dense2_tables.clamp_(-1, 1)

    @torch.no_grad()
    def discrete_state(self):
        state = super().discrete_state()
        selected = self.dense2_input_candidates.gather(
            -1, self.dense2_mapping_logits.argmax(-1, keepdim=True)).squeeze(-1)
        state.update(student_model=self.model_name, head_output_widths=(self.nodes, self.nodes),
                     group_size=self.group_size, groupsum_source="dense2",
                     dense2={"connections": selected.cpu(), "truth_tables": (self.dense2_tables >= 0).cpu(),
                             "address_pin0_is_lsb": True})
        return state


class TinyConv2DenseWide10LUT6Student(TinyConv2DenseLUT6Student):
    """Q2+LBP2 -> conv32/64 -> 8,800 LUT6 -> 8,800 LUT6 -> GroupSum."""

    nodes = 8800
    group_size = 880
    model_name = "tinyconv_q2_lbp2_dense2_wide10"


class TinyConvQ2LBP2Dense2_6000LUT6Student(TinyConvLUT6Student):
    """Q2+LBP2 -> conv32/64 -> 6,000 LUT6 -> 6,000 LUT6 -> GroupSum."""

    nodes = 6000
    group_size = 600
    model_name = "tinyconv_q2_lbp2_dense2_6000"

    def __init__(self, *, thresholds, mapping_candidates=16, mapping_temperature=1.0,
                 student_tau=40.0, node_chunk=256):
        super().__init__(thresholds=thresholds, mapping_candidates=mapping_candidates,
                         mapping_temperature=mapping_temperature, student_tau=student_tau,
                         node_chunk=node_chunk)
        self.register_buffer(
            "dense2_input_candidates",
            _candidate_pool(self.nodes, self.nodes, mapping_candidates),
        )
        self.dense2_mapping_logits = nn.Parameter(
            torch.randn(self.nodes, self.pins, mapping_candidates) * 0.01
        )
        self.dense2_tables = nn.Parameter(torch.rand(self.nodes, 64) * 2 - 1)

    def forward(self, images):
        first = _lut6_nodes(
            self.encode(images), self.input_candidates, self.mapping_logits,
            self.tables, temperature=self.mapping_temperature,
            chunk=self.node_chunk, training=self.training,
        )
        second = _lut6_nodes(
            first, self.dense2_input_candidates, self.dense2_mapping_logits,
            self.dense2_tables, temperature=self.mapping_temperature,
            chunk=self.node_chunk, training=self.training,
        )
        return second.reshape(-1, self.classes, self.group_size).sum(-1) / self.student_tau

    @torch.no_grad()
    def clamp_tables(self):
        super().clamp_tables()
        self.dense2_tables.clamp_(-1, 1)

    @torch.no_grad()
    def discrete_state(self):
        state = super().discrete_state()
        dense2 = self.dense2_input_candidates.gather(
            -1, self.dense2_mapping_logits.argmax(-1, keepdim=True)
        ).squeeze(-1)
        state.update(
            student_model=self.model_name,
            head_output_widths=(self.nodes, self.nodes),
            groupsum_source="dense2",
            group_size=self.group_size,
            dense2={
                "connections": dense2.cpu(),
                "truth_tables": (self.dense2_tables >= 0).cpu(),
                "address_pin0_is_lsb": True,
            },
        )
        return state


class TinyConvQ2OneConvDense2LUT6Student(LUT6Student):
    """Q2-only -> one LUT6 convolution -> two 8,000-node dense layers."""

    nodes = 8000
    group_size = 800
    model_name = "tinyconv_q2_1conv_dense2"
    encoder_mode = "quantile"
    encoder_quantile_bits = 2
    encoder_lbp_bits = 0
    input_bits = 32 * 16 * 16

    def _build_encoder(self, thresholds):
        from ..model import HybridFeatureEncoder

        return HybridFeatureEncoder(
            mode="quantile", thresholds=thresholds,
            num_quantile_bits=2, num_sine_bits=0,
        )

    def __init__(self, *, thresholds, mapping_candidates=16, mapping_temperature=1.0,
                 student_tau=40.0, node_chunk=256):
        super().__init__(mapping_candidates=mapping_candidates,
                         mapping_temperature=mapping_temperature,
                         student_tau=student_tau, node_chunk=node_chunk)
        del self.thresholds
        self.encoder = self._build_encoder(thresholds)
        options = dict(mapping_candidates=mapping_candidates,
                       mapping_temperature=mapping_temperature,
                       node_chunk=node_chunk)
        conv_options = dict(options)
        conv_options["mapping_candidates"] = min(mapping_candidates, (6 * 9) // 6)
        self.conv1 = _LUT6Conv2d(self.encoder.output_channels, 32, **conv_options)
        self.register_buffer(
            "dense2_input_candidates",
            _candidate_pool(self.nodes, self.nodes, mapping_candidates),
        )
        self.dense2_mapping_logits = nn.Parameter(
            torch.randn(self.nodes, self.pins, mapping_candidates) * 0.01
        )
        self.dense2_tables = nn.Parameter(torch.rand(self.nodes, 64) * 2 - 1)

    def encode(self, images):
        if images.ndim != 4 or tuple(images.shape[1:]) != (3, 32, 32):
            raise ValueError("Expected RGB CIFAR images [batch,3,32,32] in [0,1].")
        return self.conv1(self.encoder(images)).flatten(1)

    def forward(self, images):
        first = _lut6_nodes(
            self.encode(images), self.input_candidates, self.mapping_logits,
            self.tables, temperature=self.mapping_temperature,
            chunk=self.node_chunk, training=self.training,
        )
        second = _lut6_nodes(
            first, self.dense2_input_candidates, self.dense2_mapping_logits,
            self.dense2_tables, temperature=self.mapping_temperature,
            chunk=self.node_chunk, training=self.training,
        )
        return second.reshape(-1, self.classes, self.group_size).sum(-1) / self.student_tau

    @torch.no_grad()
    def clamp_tables(self):
        LUT6Student.clamp_tables(self)
        self.conv1.tables.clamp_(-1, 1)
        self.dense2_tables.clamp_(-1, 1)

    @torch.no_grad()
    def discrete_state(self):
        dense1 = self.input_candidates.gather(
            -1, self.mapping_logits.argmax(-1, keepdim=True)
        ).squeeze(-1)
        dense2 = self.dense2_input_candidates.gather(
            -1, self.dense2_mapping_logits.argmax(-1, keepdim=True)
        ).squeeze(-1)
        return {
            "student_model": self.model_name,
            "encoder": {
                "mode": self.encoder_mode,
                "quantile_bits": self.encoder_quantile_bits,
                "lbp_bits": self.encoder_lbp_bits,
                "state": {key: value.cpu().clone() for key, value in self.encoder.state_dict().items()},
            },
            "conv1": self.conv1.discrete_state(),
            "connections": dense1.cpu(),
            "truth_tables": (self.tables >= 0).cpu(),
            "dense2": {
                "connections": dense2.cpu(),
                "truth_tables": (self.dense2_tables >= 0).cpu(),
                "address_pin0_is_lsb": True,
            },
            "flatten_order": "channel,row,column",
            "head_input_shape": (32, 16, 16),
            "head_output_widths": (self.nodes, self.nodes),
            "groupsum_source": "dense2",
            "student_tau": self.student_tau,
            "group_size": self.group_size,
            "address_pin0_is_lsb": True,
        }


class TinyConvQ2Wide10Dense2LUT6Student(LUT6Student):
    """Q2-only -> conv35/70 -> 8,800 LUT6 -> 8,800 LUT6 -> GroupSum."""

    nodes = 8800
    input_bits = 70 * 8 * 8
    group_size = 880
    conv1_channels = 35
    conv2_channels = 70
    model_name = "tinyconv_q2_wide10_dense2"
    encoder_mode = "quantile"
    encoder_quantile_bits = 2
    encoder_lbp_bits = 0

    def _build_encoder(self, thresholds):
        from ..model import HybridFeatureEncoder

        return HybridFeatureEncoder(
            mode="quantile", thresholds=thresholds,
            num_quantile_bits=2, num_sine_bits=0,
        )

    def __init__(self, *, thresholds, mapping_candidates=16, mapping_temperature=1.0,
                 student_tau=40.0, node_chunk=256):
        super().__init__(mapping_candidates=mapping_candidates,
                         mapping_temperature=mapping_temperature,
                         student_tau=student_tau, node_chunk=node_chunk)
        del self.thresholds
        self.encoder = self._build_encoder(thresholds)
        options = dict(mapping_candidates=mapping_candidates,
                       mapping_temperature=mapping_temperature,
                       node_chunk=node_chunk)
        conv1_options = dict(options)
        conv1_options["mapping_candidates"] = min(mapping_candidates, (6 * 9) // 6)
        self.conv1 = _LUT6Conv2d(self.encoder.output_channels, self.conv1_channels, **conv1_options)
        self.conv2 = _LUT6Conv2d(self.conv1_channels, self.conv2_channels, **options)
        self.register_buffer(
            "dense2_input_candidates",
            _candidate_pool(self.nodes, self.nodes, mapping_candidates),
        )
        self.dense2_mapping_logits = nn.Parameter(
            torch.randn(self.nodes, self.pins, mapping_candidates) * 0.01
        )
        self.dense2_tables = nn.Parameter(torch.rand(self.nodes, 64) * 2 - 1)

    def encode(self, images):
        if images.ndim != 4 or tuple(images.shape[1:]) != (3, 32, 32):
            raise ValueError("Expected RGB CIFAR images [batch,3,32,32] in [0,1].")
        return self.conv2(self.conv1(self.encoder(images))).flatten(1)

    def forward(self, images):
        first = _lut6_nodes(
            self.encode(images), self.input_candidates, self.mapping_logits, self.tables,
            temperature=self.mapping_temperature, chunk=self.node_chunk, training=self.training,
        )
        second = _lut6_nodes(
            first, self.dense2_input_candidates, self.dense2_mapping_logits, self.dense2_tables,
            temperature=self.mapping_temperature, chunk=self.node_chunk, training=self.training,
        )
        return second.reshape(-1, self.classes, self.group_size).sum(-1) / self.student_tau

    @torch.no_grad()
    def clamp_tables(self):
        super().clamp_tables()
        self.conv1.tables.clamp_(-1, 1)
        self.conv2.tables.clamp_(-1, 1)
        self.dense2_tables.clamp_(-1, 1)

    @torch.no_grad()
    def discrete_state(self):
        dense1 = self.input_candidates.gather(
            -1, self.mapping_logits.argmax(-1, keepdim=True)
        ).squeeze(-1)
        dense2 = self.dense2_input_candidates.gather(
            -1, self.dense2_mapping_logits.argmax(-1, keepdim=True)
        ).squeeze(-1)
        return {
            "student_model": self.model_name,
            "encoder": {
                "mode": self.encoder_mode,
                "quantile_bits": self.encoder_quantile_bits,
                "lbp_bits": self.encoder_lbp_bits,
                "state": {key: value.cpu().clone() for key, value in self.encoder.state_dict().items()},
            },
            "conv1": self.conv1.discrete_state(),
            "conv2": self.conv2.discrete_state(),
            "connections": dense1.cpu(),
            "truth_tables": (self.tables >= 0).cpu(),
            "dense2": {
                "connections": dense2.cpu(),
                "truth_tables": (self.dense2_tables >= 0).cpu(),
                "address_pin0_is_lsb": True,
            },
            "flatten_order": "channel,row,column",
            "head_input_shape": (self.conv2_channels, 8, 8),
            "head_output_widths": (self.nodes, self.nodes),
            "groupsum_source": "dense2",
            "student_tau": self.student_tau,
            "group_size": self.group_size,
            "address_pin0_is_lsb": True,
        }


class TinyConvQ2Compact20Dense2LUT6Student(TinyConvQ2Wide10Dense2LUT6Student):
    """Q2-only, with every trainable layer 20% narrower than the wide variant."""

    nodes = 7040
    input_bits = 56 * 8 * 8
    group_size = 704
    conv1_channels = 28
    conv2_channels = 56
    model_name = "tinyconv_q2_compact20_dense2"


class TinyConvQ2Wide10Dense2FourConvLUT6Student(TinyConvQ2Wide10Dense2LUT6Student):
    """Q2-only -> four strided LUT6 convolutions -> two 8,800-node dense layers."""

    model_name = "tinyconv_q2_wide10_dense2_4conv"
    conv_channels = (35, 70, 105, 140)
    input_bits = conv_channels[-1] * 2 * 2

    def __init__(self, *, thresholds, mapping_candidates=16, mapping_temperature=1.0,
                 student_tau=40.0, node_chunk=256):
        LUT6Student.__init__(
            self, mapping_candidates=mapping_candidates,
            mapping_temperature=mapping_temperature,
            student_tau=student_tau, node_chunk=node_chunk,
        )
        self.encoder = self._build_encoder(thresholds)
        options = dict(mapping_candidates=mapping_candidates,
                       mapping_temperature=mapping_temperature,
                       node_chunk=node_chunk)
        convs = []
        in_channels = self.encoder.output_channels
        for index, out_channels in enumerate(self.conv_channels, start=1):
            conv_options = dict(options)
            if index == 1:
                conv_options["mapping_candidates"] = min(mapping_candidates, (6 * 9) // 6)
            conv = _LUT6Conv2d(in_channels, out_channels, **conv_options)
            setattr(self, f"conv{index}", conv)
            convs.append(conv)
            in_channels = out_channels
        self._convs = nn.ModuleList(convs)
        self.head_input_bits = in_channels * 2 * 2
        self.register_buffer("dense2_input_candidates", _candidate_pool(
            self.nodes, self.nodes, mapping_candidates,
        ))
        self.dense2_mapping_logits = nn.Parameter(
            torch.randn(self.nodes, self.pins, mapping_candidates) * 0.01
        )
        self.dense2_tables = nn.Parameter(torch.rand(self.nodes, 64) * 2 - 1)

    def encode(self, images):
        if images.ndim != 4 or tuple(images.shape[1:]) != (3, 32, 32):
            raise ValueError("Expected RGB CIFAR images [batch,3,32,32] in [0,1].")
        features = self.encoder(images)
        for conv in self._convs:
            features = conv(features)
        return features.flatten(1)

    def forward(self, images):
        first = _lut6_nodes(
            self.encode(images), self.input_candidates, self.mapping_logits,
            self.tables, temperature=self.mapping_temperature,
            chunk=self.node_chunk, training=self.training,
        )
        second = _lut6_nodes(
            first, self.dense2_input_candidates, self.dense2_mapping_logits,
            self.dense2_tables, temperature=self.mapping_temperature,
            chunk=self.node_chunk, training=self.training,
        )
        return second.reshape(-1, self.classes, self.group_size).sum(-1) / self.student_tau

    @torch.no_grad()
    def clamp_tables(self):
        LUT6Student.clamp_tables(self)
        for conv in self._convs:
            conv.tables.clamp_(-1, 1)
        self.dense2_tables.clamp_(-1, 1)

    @torch.no_grad()
    def discrete_state(self):
        state = {
            "student_model": self.model_name,
            "encoder": {
                "mode": self.encoder_mode,
                "quantile_bits": self.encoder_quantile_bits,
                "lbp_bits": self.encoder_lbp_bits,
                "state": {key: value.cpu().clone() for key, value in self.encoder.state_dict().items()},
            },
            "flatten_order": "channel,row,column",
            "head_input_shape": (self.conv_channels[-1], 2, 2),
            "head_output_widths": (self.nodes, self.nodes),
            "groupsum_source": "dense2",
            "student_tau": self.student_tau,
            "group_size": self.group_size,
            "address_pin0_is_lsb": True,
        }
        for index, conv in enumerate(self._convs, start=1):
            state[f"conv{index}"] = conv.discrete_state()
        state["connections"] = self.input_candidates.gather(
            -1, self.mapping_logits.argmax(-1, keepdim=True)
        ).squeeze(-1).cpu()
        state["truth_tables"] = (self.tables >= 0).cpu()
        state["dense2"] = {
            "connections": self.dense2_input_candidates.gather(
                -1, self.dense2_mapping_logits.argmax(-1, keepdim=True)
            ).squeeze(-1).cpu(),
            "truth_tables": (self.dense2_tables >= 0).cpu(),
            "address_pin0_is_lsb": True,
        }
        return state


class TinyConvLBP2Wide10Dense2LUT6Student(TinyConvQ2Wide10Dense2LUT6Student):
    """LBP2-only -> conv35/70 -> 8,800 LUT6 -> 8,800 LUT6 -> GroupSum."""

    model_name = "tinyconv_lbp2_wide10_dense2"
    encoder_mode = "lbp2"
    encoder_quantile_bits = 0
    encoder_lbp_bits = 2

    def _build_encoder(self, thresholds):
        return _LBP2Encoder()
