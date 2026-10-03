"""Shared logic model used by every comparison arm."""

import math

import torch

from torchlogix.connections import FixedConvConnections
from torchlogix.layers import (
    ClassRoutedLogicDense,
    GroupSum,
    LogicConv2d,
    LogicDense,
    ModifiedLogicDense,
    OrPooling2d,
)
from torchlogix.models.conv import (
    ClgnCifar10_deepconv,
    ClgnCifar10Large,
    ClgnCifar10Small,
)

from .wire_erasing import BooleanWireErasure

        
class StraightThroughBinarize(torch.autograd.Function):
    """Hard zero-threshold binarization with an identity backward pass."""

    @staticmethod
    def forward(ctx, values: torch.Tensor) -> torch.Tensor:
        del ctx
        return (values > 0.0).to(values.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        del ctx
        return grad_output.clone()

class HybridFeatureEncoder(torch.nn.Module):
    """
    Encode RGB CIFAR inputs into binary quantile and/or sinusoidal bitplanes.

    Supported modes
    ---------------
    fixed:
        Legacy threshold/thermometer encoding.

    quantile:
        Quantile threshold encoding only.

    sinusoidal:
        Sinusoidal binary features only.

    hybrid:
        Quantile bits followed by sinusoidal bits for each RGB channel.

    Hybrid output ordering
    ----------------------
    For Q quantile bits and S sine bits:

        R_q1 ... R_qQ, R_s1 ... R_sS,
        G_q1 ... G_qQ, G_s1 ... G_sS,
        B_q1 ... B_qQ, B_s1 ... B_sS

    Important changes from the previous implementation
    ---------------------------------------------------
    1. Sine frequencies are interpreted as cycles over x in [0, 1].
       The actual function is:

           sin(2*pi*frequency*x + phase)

       rather than:

           sin(frequency*x + phase)

    2. Random sine frequencies are prevented from initializing arbitrarily
       close to zero. Near-zero frequencies frequently create constant binary
       bitplanes after thresholding.

    3. Sine phases are spread across [0, pi) rather than all being completely
       independent. This reduces redundant/complementary sine bitplanes.

    4. Sine frequencies/phases can optionally be trainable. The existing
       StraightThroughBinarize then propagates gradients into them.

    5. Diagnostic helpers are included for inspecting bit activation rates.

    Notes
    -----
    ``sigma`` now controls the standard deviation of sine FREQUENCIES measured
    in cycles across the normalized pixel interval [0, 1].

    With sigma=1.0, a frequency near 1 means approximately one complete sine
    cycle as the pixel value moves from 0 to 1.
    """

    MODES = ("fixed", "quantile", "sinusoidal", "hybrid")

    def __init__(
        self,
        mode: str = "hybrid",
        *,
        thresholds: torch.Tensor | None = None,
        num_quantile_bits: int = 2,
        num_sine_bits: int = 2,
        sigma: float = 1.0,
        sine_seed: int = 0,
        learnable_sine: bool = False,
        min_sine_frequency: float = 0.5,
    ):
        super().__init__()

        # ------------------------------------------------------------
        # Validate common configuration
        # ------------------------------------------------------------
        if mode not in self.MODES:
            raise ValueError(
                f"Unknown encoder mode {mode!r}; "
                f"choose from {self.MODES}."
            )

        if sigma <= 0:
            raise ValueError(
                f"sigma must be positive, got {sigma}."
            )

        if min_sine_frequency < 0:
            raise ValueError(
                "min_sine_frequency must be non-negative, "
                f"got {min_sine_frequency}."
            )

        uses_quantiles = mode in {"fixed", "quantile", "hybrid"}
        uses_sine = mode in {"sinusoidal", "hybrid"}

        self.mode = mode
        self.sigma = float(sigma)
        self.sine_seed = int(sine_seed)
        self.learnable_sine = bool(learnable_sine)
        self.min_sine_frequency = float(min_sine_frequency)

        # ============================================================
        # Quantile / threshold branch
        # ============================================================
        if uses_quantiles:
            if num_quantile_bits <= 0:
                raise ValueError(
                    "num_quantile_bits must be positive when "
                    f"mode={mode!r}."
                )

            if thresholds is None:
                raise ValueError(
                    f"Encoder mode {mode!r} requires RGB thresholds."
                )

            threshold_tensor = torch.as_tensor(
                thresholds,
                dtype=torch.float32,
            ).detach().clone()

            expected_shape = (3, num_quantile_bits)

            if tuple(threshold_tensor.shape) != expected_shape:
                raise ValueError(
                    f"Expected thresholds with shape {expected_shape}, "
                    f"got {tuple(threshold_tensor.shape)}."
                )

            if torch.any(
                (threshold_tensor < 0.0)
                | (threshold_tensor > 1.0)
            ):
                raise ValueError(
                    "Input thresholds must lie in [0, 1]."
                )

            # Thermometer thresholds must remain ordered.
            if num_quantile_bits > 1:
                if torch.any(
                    threshold_tensor[:, 1:]
                    < threshold_tensor[:, :-1]
                ):
                    raise ValueError(
                        "Thresholds must be nondecreasing "
                        "within each RGB channel."
                    )

        else:
            num_quantile_bits = 0
            threshold_tensor = torch.empty(
                (3, 0),
                dtype=torch.float32,
            )

        self._num_quantile_bits = int(num_quantile_bits)

        # Quantile thresholds remain fixed.
        self.register_buffer(
            "thresholds",
            threshold_tensor,
        )

        # ============================================================
        # Sinusoidal branch
        # ============================================================
        if uses_sine:
            if num_sine_bits <= 0:
                raise ValueError(
                    "num_sine_bits must be positive when "
                    f"mode={mode!r}."
                )

            generator = torch.Generator(
                device="cpu"
            ).manual_seed(self.sine_seed)

            # --------------------------------------------------------
            # Frequencies
            #
            # These are frequencies in CYCLES over x in [0,1].
            #
            # Forward later evaluates:
            #
            #   sin(2*pi*f*x + phase)
            #
            # Prevent frequencies close to zero because those often
            # produce almost-constant binary bitplanes.
            # --------------------------------------------------------
            frequencies = self.sigma * torch.randn(
                (3, num_sine_bits),
                generator=generator,
                dtype=torch.float32,
            )

            if self.min_sine_frequency > 0:
                signs = torch.where(
                    frequencies < 0,
                    -torch.ones_like(frequencies),
                    torch.ones_like(frequencies),
                )

                magnitudes = frequencies.abs().clamp_min(
                    self.min_sine_frequency
                )

                frequencies = signs * magnitudes

            # --------------------------------------------------------
            # Phases
            #
            # Give each RGB channel a random base phase, but distribute
            # multiple sine bits over a half-period.
            #
            # We use [0, pi), rather than [0, 2*pi), because phases
            # separated by exactly pi produce complementary binary
            # features:
            #
            #   sign(sin(theta + pi)) = NOT sign(sin(theta))
            #
            # Avoiding exact complements gives the logic model more
            # diverse inputs.
            # --------------------------------------------------------
            base_phase = (
                2.0
                * math.pi
                * torch.rand(
                    (3, 1),
                    generator=generator,
                    dtype=torch.float32,
                )
            )

            if num_sine_bits == 1:
                phase_offsets = torch.zeros(
                    (1, 1),
                    dtype=torch.float32,
                )
            else:
                phase_offsets = (
                    torch.arange(
                        num_sine_bits,
                        dtype=torch.float32,
                    ).view(1, num_sine_bits)
                    * (math.pi / num_sine_bits)
                )

            phases = torch.remainder(
                base_phase + phase_offsets,
                2.0 * math.pi,
            )

        else:
            num_sine_bits = 0

            frequencies = torch.empty(
                (3, 0),
                dtype=torch.float32,
            )

            phases = torch.empty(
                (3, 0),
                dtype=torch.float32,
            )

        self._num_sine_bits = int(num_sine_bits)

        # ------------------------------------------------------------
        # Optionally allow the sine encoder itself to learn.
        #
        # StraightThroughBinarize has an identity backward pass, so
        # gradients can reach these parameters through:
        #
        #     binary <- sin(...) <- frequency / phase
        # ------------------------------------------------------------
        if uses_sine and self.learnable_sine:
            self.sine_weights = torch.nn.Parameter(
                frequencies
            )
            self.sine_phases = torch.nn.Parameter(
                phases
            )
        else:
            self.register_buffer(
                "sine_weights",
                frequencies,
            )
            self.register_buffer(
                "sine_phases",
                phases,
            )

    # ================================================================
    # Properties
    # ================================================================

    @property
    def num_quantile_bits(self) -> int:
        return self._num_quantile_bits

    @property
    def num_sine_bits(self) -> int:
        return self._num_sine_bits

    @property
    def num_bits(self) -> int:
        """
        Number of encoded bits PER RGB channel.
        """
        return (
            self.num_quantile_bits
            + self.num_sine_bits
        )

    @property
    def output_channels(self) -> int:
        """
        Total encoded channels presented to the first logic layer.
        """
        return 3 * self.num_bits

    @property
    def channel_order(self) -> str:
        if self.mode == "hybrid":
            return (
                "per RGB channel: quantile bits, then sine bits"
            )

        if self.num_quantile_bits:
            return (
                "R quantile bits, "
                "G quantile bits, "
                "B quantile bits"
            )

        return (
            "R sine bits, "
            "G sine bits, "
            "B sine bits"
        )

    # ================================================================
    # Encoding helpers
    # ================================================================

    def _encode_quantiles(
        self,
        expanded: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Encode quantile/thermometer features.

        Input
        -----
        expanded:
            [B, 3, 1, H, W]

        Output
        ------
            [B, 3, Q, H, W]
        """

        thresholds = self.thresholds.view(
            1,
            3,
            self.num_quantile_bits,
            1,
            1,
        )

        return (
            expanded >= thresholds
        ).to(dtype)

    def _encode_sine(
        self,
        expanded: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode sinusoidal binary features.

        Input
        -----
        expanded:
            [B, 3, 1, H, W]

        Output
        ------
            [B, 3, S, H, W]
        """

        frequencies = self.sine_weights.view(
            1,
            3,
            self.num_sine_bits,
            1,
            1,
        )

        phases = self.sine_phases.view(
            1,
            3,
            self.num_sine_bits,
            1,
            1,
        )

        # ------------------------------------------------------------
        # CRITICAL CHANGE
        #
        # Previous:
        #
        #   sin(weight * x + phase)
        #
        # New:
        #
        #   sin(2*pi*frequency*x + phase)
        #
        # frequency=1 therefore means one complete cycle over x=[0,1].
        # ------------------------------------------------------------
        sine_values = torch.sin(
            (2.0 * math.pi * frequencies * expanded)
            + phases
        )

        return StraightThroughBinarize.apply(
            sine_values
        )

    # ================================================================
    # Forward
    # ================================================================

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError(
                "Expected continuous RGB input with shape "
                "[B,3,H,W], "
                f"got {tuple(x.shape)}."
            )

        # We intentionally do not clamp x here.
        #
        # Your CIFAR preprocessing should supply values in [0,1].
        # Silently clamping here could hide an incorrect transform.
        expanded = x.unsqueeze(2)

        # Shape:
        #
        #   [B, 3, 1, H, W]
        #
        # We build features along dimension 2 so hybrid ordering stays:
        #
        #   channel:
        #       quantile bits -> sine bits
        feature_groups = []

        if self.num_quantile_bits:
            quantile_bits = self._encode_quantiles(
                expanded,
                x.dtype,
            )

            feature_groups.append(
                quantile_bits
            )

        if self.num_sine_bits:
            sine_bits = self._encode_sine(
                expanded
            )

            feature_groups.append(
                sine_bits
            )

        if not feature_groups:
            raise RuntimeError(
                "Encoder produced no feature groups."
            )

        # ------------------------------------------------------------
        # Hybrid:
        #
        # [B,3,Q,H,W]
        #      +
        # [B,3,S,H,W]
        #
        # ->
        #
        # [B,3,Q+S,H,W]
        #
        # Then reshape RGB and bit dimensions together.
        # ------------------------------------------------------------
        encoded = torch.cat(
            feature_groups,
            dim=2,
        )

        encoded = encoded.reshape(
            x.shape[0],
            self.output_channels,
            x.shape[2],
            x.shape[3],
        )

        return encoded

    # ================================================================
    # Diagnostics
    # ================================================================

    @torch.no_grad()
    def activation_rates(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return the fraction of active pixels for each encoded bit.

        Returns
        -------
        Tensor with shape:

            [3, num_bits]

        For hybrid mode, columns are:

            q1 ... qQ, s1 ... sS

        for each RGB row.

        Healthy binary features are generally not extremely close to
        0 or 1 across the full training distribution.

        Example
        -------
        rates = encoder.activation_rates(images)

        print("R:", rates[0])
        print("G:", rates[1])
        print("B:", rates[2])
        """

        encoded = self.forward(x)

        batch_size, _, height, width = encoded.shape

        encoded = encoded.reshape(
            batch_size,
            3,
            self.num_bits,
            height,
            width,
        )

        return encoded.float().mean(
            dim=(0, 3, 4)
        )

    @torch.no_grad()
    def detailed_activation_report(
        self,
        x: torch.Tensor,
    ) -> dict[str, float]:
        """
        Human-readable activation report for debugging hybrid features.
        """

        rates = self.activation_rates(x)

        report = {}

        for rgb_index, rgb_name in enumerate(
            ("R", "G", "B")
        ):
            bit_index = 0

            for q in range(
                self.num_quantile_bits
            ):
                report[
                    f"{rgb_name}_quantile_{q + 1}"
                ] = rates[
                    rgb_index,
                    bit_index,
                ].item()

                bit_index += 1

            for s in range(
                self.num_sine_bits
            ):
                report[
                    f"{rgb_name}_sine_{s + 1}"
                ] = rates[
                    rgb_index,
                    bit_index,
                ].item()

                bit_index += 1

        return report

    # ================================================================
    # Representation
    # ================================================================

    def extra_repr(self) -> str:
        return (
            f"mode={self.mode!r}, "
            f"quantile_bits={self.num_quantile_bits}, "
            f"sine_bits={self.num_sine_bits}, "
            f"output_channels={self.output_channels}, "
            f"sigma={self.sigma:g}, "
            f"min_sine_frequency={self.min_sine_frequency:g}, "
            f"sine_seed={self.sine_seed}, "
            f"learnable_sine={self.learnable_sine}"
        )

class QuantileThermometerEncoder(HybridFeatureEncoder):
    """Encode continuous RGB values with fixed per-channel thresholds.

    Thresholds have shape ``[3, num_bits]`` and are stored as a model buffer.
    Output channels are ordered as all R bits, then G bits, then B bits.
    """

    def __init__(self, thresholds: torch.Tensor):
        thresholds = torch.as_tensor(thresholds, dtype=torch.float32).detach().clone()
        if thresholds.ndim != 2 or thresholds.shape[0] != 3:
            raise ValueError(
                "Expected RGB thresholds with shape [3,num_bits], "
                f"got {tuple(thresholds.shape)}."
            )
        super().__init__(
            mode="quantile",
            thresholds=thresholds,
            num_quantile_bits=thresholds.shape[1],
            num_sine_bits=0,
        )


def _build_cifar10_encoder(
    input_thresholds: torch.Tensor | None,
    *,
    input_encoding: str | None = None,
    num_quantile_bits: int = 2,
    num_sine_bits: int = 2,
    sine_sigma: float = 1.0,
    sine_seed: int = 0,
) -> HybridFeatureEncoder:
    """Build the selected encoder with a fixed 1-bit legacy fallback."""

    if input_encoding is None:
        input_encoding = "fixed" if input_thresholds is None else "quantile"
    if input_encoding == "fixed":
        if input_thresholds is None:
            input_thresholds = torch.full((3, 1), 0.5, dtype=torch.float32)
        return QuantileThermometerEncoder(input_thresholds)
    if input_encoding == "quantile":
        if input_thresholds is None:
            raise ValueError("Quantile encoding requires training-set thresholds.")
        return QuantileThermometerEncoder(input_thresholds)
    if input_encoding not in {"sinusoidal", "hybrid"}:
        raise ValueError(f"Unsupported CIFAR input encoding {input_encoding!r}.")
    return HybridFeatureEncoder(
        mode=input_encoding,
        thresholds=input_thresholds,
        num_quantile_bits=num_quantile_bits,
        num_sine_bits=num_sine_bits,
        sigma=sine_sigma,
        sine_seed=sine_seed,
    )


def build_model(device: torch.device, *, parametrization: str) -> torch.nn.Module:
    """Build the common architecture with the selected LUT parametrization."""

    logic_options = {
        "lut_rank": 2,
        "parametrization": parametrization,
        "parametrization_kwargs": {
            "forward_sampling": "soft",
            "temperature": 1.0,
        },
        "device": str(device),
    }

    return torch.nn.Sequential(
        LogicConv2d(
            in_dim=28,
            channels=1,
            num_kernels=16,
            tree_depth=2,
            receptive_field_size=3,
            **logic_options,
        ),
        OrPooling2d(kernel_size=2, stride=2, padding=0),
        torch.nn.Flatten(),
        LogicDense(16 * 13 * 13, 4_000, **logic_options),
        LogicDense(4_000, 4_000, **logic_options),
        GroupSum(k=10, tau=8),
    ).to(device)


def build_model_linear_head(
    device: torch.device,
    *,
    parametrization: str,
) -> torch.nn.Module:
    """Build the baseline architecture with a trainable linear classifier."""

    logic_options = {
        "lut_rank": 2,
        "parametrization": parametrization,
        "parametrization_kwargs": {
            "forward_sampling": "soft",
            "temperature": 1.0,
        },
        "device": str(device),
    }

    return torch.nn.Sequential(
        LogicConv2d(
            in_dim=28,
            channels=1,
            num_kernels=16,
            tree_depth=2,
            receptive_field_size=3,
            **logic_options,
        ),
        OrPooling2d(kernel_size=2, stride=2, padding=0),
        torch.nn.Flatten(),
        LogicDense(16 * 13 * 13, 4_000, **logic_options),
        LogicDense(4_000, 4_000, **logic_options),
        torch.nn.Linear(4_000, 10),
    ).to(device)

def build_model_cifar10_org(
    device: torch.device,
    *,
    parametrization: str,
    input_thresholds: torch.Tensor | None = None,
    input_encoding: str | None = None,
    num_quantile_bits: int = 2,
    num_sine_bits: int = 2,
    sine_sigma: float = 1.0,
    sine_seed: int = 0,
) -> torch.nn.Module:
    """Build the original GroupSum architecture for encoded RGB CIFAR-10."""

    width = 16_000
    num_kernels = 64
    encoder = _build_cifar10_encoder(
        input_thresholds,
        input_encoding=input_encoding,
        num_quantile_bits=num_quantile_bits,
        num_sine_bits=num_sine_bits,
        sine_sigma=sine_sigma,
        sine_seed=sine_seed,
    )

    logic_options = {
        "lut_rank": 4,
        "parametrization": parametrization,
        "parametrization_kwargs": {
            "forward_sampling": "soft",
            "temperature": 1.0,
        },
        "device": str(device),
    }

    return torch.nn.Sequential(
        encoder,
        LogicConv2d(
            in_dim=32,
            channels=encoder.output_channels,
            num_kernels=num_kernels,
            tree_depth=2,
            receptive_field_size=3,
            **logic_options,
        ),
        OrPooling2d(kernel_size=2, stride=2, padding=0),
        torch.nn.Flatten(),
        LogicDense(num_kernels * 15 * 15, width, **logic_options),
        LogicDense(width, width, **logic_options),
        GroupSum(k=10, tau=8),
    ).to(device)

#under training
#remove_smallmodel name from end
def build_model_cifar10_smallmodel(
    device: torch.device,
    *,
    parametrization: str,
    input_thresholds: torch.Tensor | None = None,
    input_encoding: str | None = None,
    num_quantile_bits: int = 2,
    num_sine_bits: int = 2,
    sine_sigma: float = 1.0,
    sine_seed: int = 0,
    group_size: int = 2,
    first_conv_init_method: str | None = None,
) -> torch.nn.Module:
    """Build CLGN-small with the shared external CIFAR encoder."""

    encoder = _build_cifar10_encoder(
        input_thresholds,
        input_encoding=input_encoding,
        num_quantile_bits=num_quantile_bits,
        num_sine_bits=num_sine_bits,
        sine_sigma=sine_sigma,
        sine_seed=sine_seed,
    )

    # ClgnCifar10Small normally fixes n_input_bits=2 and creates its own
    # binarizer. This local specialization keeps every later CLGN-small layer
    # unchanged while sizing only the first convolution for our encoder.
    encoded_model_type = type(
        "EncodedClgnCifar10Small",
        (ClgnCifar10Small,),
        {"n_input_bits": encoder.num_bits,
        "k_num": 36,# was 32 # 36 is best 
        "group_size": group_size,
        "group_size_input": 1,}, 
    )
    model = encoded_model_type(
        # The dummy CLGN binarizer ignores these values but its constructor
        # validates the last dimension against the total encoded bit count.
        thresholds=encoder.thresholds.new_empty((3, encoder.num_bits)),
        binarization="dummy",
        binarization_kwargs={},
        connections_kwargs={"init_method": "random-unique"},
        lut_rank=2,
        parametrization=parametrization,
        parametrization_kwargs={
            "forward_sampling": "soft",
            "temperature": 1.0,
        },
        device=str(device),
    )
    model[0] = encoder
    if first_conv_init_method is not None:
        first_conv = next(
            module for module in model if isinstance(module, LogicConv2d)
        )
        first_conv.connections = FixedConvConnections(
            in_dim=first_conv.in_dim,
            channels=encoder.output_channels,
            num_kernels=first_conv.num_kernels,
            tree_depth=first_conv.tree_depth,
            receptive_field_size=first_conv.receptive_field_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            conv_dimension=first_conv.conv_dimension,
            lut_rank=first_conv.lut_rank,
            device=str(device),
            init_method=first_conv_init_method,
            channel_group_size=None,
        )
    first_dense_index = next(
        index
        for index, module in enumerate(model)
        if isinstance(module, LogicDense)
    )
    model.insert(first_dense_index + 1, BooleanWireErasure(p=0.0))
    return model.to(device)


def build_model_cifar10_small_modifieddense(
    device: torch.device,
    *,
    parametrization: str,
    input_thresholds: torch.Tensor | None = None,
    input_encoding: str | None = None,
    num_quantile_bits: int = 2,
    num_sine_bits: int = 2,
    sine_sigma: float = 1.0,
    sine_seed: int = 0,
) -> torch.nn.Module:
    """CLGN-small with a dual-path first shrinking dense layer."""

    model = build_model_cifar10_smallmodel(
        device,
        parametrization=parametrization,
        input_thresholds=input_thresholds,
        input_encoding=input_encoding,
        num_quantile_bits=num_quantile_bits,
        num_sine_bits=num_sine_bits,
        sine_sigma=sine_sigma,
        sine_seed=sine_seed,
    )

    dense_indices = [
        index
        for index, module in enumerate(model)
        if isinstance(module, LogicDense)
    ]
    first_shrinking_index = dense_indices[1] #1
    original = model[first_shrinking_index]
    model[first_shrinking_index] = ModifiedLogicDense(
        in_dim=original.in_dim,
        out_dim=original.out_dim,
        device=str(device),
        grad_factor=original.grad_factor,
        lut_rank=original.lut_rank,
        parametrization=parametrization,
        parametrization_kwargs={
            "forward_sampling": "soft",
            "temperature": 1.0,
        },
        connections="fixed",
        connections_kwargs={"init_method": "random-unique"},
    )
    return model.to(device)


def build_model_cifar10_small_classrouted(
    device: torch.device,
    *,
    parametrization: str,
    input_thresholds: torch.Tensor | None = None,
    input_encoding: str | None = None,
    num_quantile_bits: int = 2,
    num_sine_bits: int = 2,
    sine_sigma: float = 1.0,
    sine_seed: int = 0,
    first_conv_init_method: str | None = None,
) -> torch.nn.Module:
    """CLGN-small with class-aligned fixed Dense connectivity."""

    model = build_model_cifar10_smallmodel(
        device,
        parametrization=parametrization,
        input_thresholds=input_thresholds,
        input_encoding=input_encoding,
        num_quantile_bits=num_quantile_bits,
        num_sine_bits=num_sine_bits,
        sine_sigma=sine_sigma,
        sine_seed=sine_seed,
        group_size=2,#was 2
        first_conv_init_method=first_conv_init_method,
    )
    dense_indices = [
        index
        for index, module in enumerate(model)
        if isinstance(module, LogicDense)
    ]
    topology_seed = torch.initial_seed()
    for layer_number, dense_index in enumerate(dense_indices):
        original = model[dense_index]
        model[dense_index] = ClassRoutedLogicDense(
            in_dim=original.in_dim,
            out_dim=original.out_dim,
            device=str(device),
            grad_factor=original.grad_factor,
            lut_rank=original.lut_rank,
            parametrization=parametrization,
            parametrization_kwargs={
                "forward_sampling": "soft",
                "temperature": 1.0,
            },
            num_classes=10,
            shared_input=layer_number == 0,
            topology_seed=topology_seed + layer_number,
        )
    return model.to(device)


def build_model_cifar10_small_classrouted_channelmixed(
    device: torch.device,
    *,
    parametrization: str,
    input_thresholds: torch.Tensor | None = None,
    input_encoding: str | None = None,
    num_quantile_bits: int = 2,
    num_sine_bits: int = 2,
    sine_sigma: float = 1.0,
    sine_seed: int = 0,
) -> torch.nn.Module:
    """CIFAR-10 class-routed model with balanced Conv1 channel mixing."""

    return build_model_cifar10_small_classrouted(
        device,
        parametrization=parametrization,
        input_thresholds=input_thresholds,
        input_encoding=input_encoding,
        num_quantile_bits=num_quantile_bits,
        num_sine_bits=num_sine_bits,
        sine_sigma=sine_sigma,
        sine_seed=sine_seed,
        first_conv_init_method="channel-balanced",
    )


def build_model_cifar100_small_classrouted(
    device: torch.device,
    *,
    parametrization: str,
    input_thresholds: torch.Tensor | None = None,
    input_encoding: str | None = None,
    num_quantile_bits: int = 2,
    num_sine_bits: int = 2,
    sine_sigma: float = 1.0,
    sine_seed: int = 0,
    first_conv_init_method: str | None = None,
) -> torch.nn.Module:
    """CLGN-small for CIFAR-100 with routing after two shared dense stages.

    The CIFAR-10 encoder and convolutional backbone are retained. Dense1 and
    Dense2 are shared 60k-gate representations. Dense3 and Dense4 produce 60k
    then 30k class-routed outputs (600 then 300 per class).
    """

    model = build_model_cifar10_smallmodel(
        device,
        parametrization=parametrization,
        input_thresholds=input_thresholds,
        input_encoding=input_encoding,
        num_quantile_bits=num_quantile_bits,
        num_sine_bits=num_sine_bits,
        sine_sigma=sine_sigma,
        sine_seed=sine_seed,
        group_size=2,
        first_conv_init_method=first_conv_init_method,
    )
    dense_indices = [
        index
        for index, module in enumerate(model)
        if isinstance(module, LogicDense)
    ]
    topology_seed = torch.initial_seed()
    shared_input_dim = model[dense_indices[0]].in_dim
    common_options = {
        "device": str(device),
        "lut_rank": 2,
        "parametrization": parametrization,
        "parametrization_kwargs": {
            "forward_sampling": "soft",
            "temperature": 1.0,
        },
    }

    # Dense1 and Dense2 remain shared across all 100 classes.
    model[dense_indices[0]] = LogicDense(
        in_dim=shared_input_dim,
        out_dim=60_000,
        connections="fixed",
        connections_kwargs={"init_method": "random-unique"},
        **common_options,
    )
    model[dense_indices[1]] = LogicDense(
        in_dim=60_000,
        out_dim=60_000,
        connections="fixed",
        connections_kwargs={"init_method": "random-unique"},
        **common_options,
    )
    model[dense_indices[2]] = ClassRoutedLogicDense(
        in_dim=60_000,
        out_dim=60_000,
        num_classes=100,
        shared_input=True,
        shared_input_subset=True,
        topology_seed=topology_seed + 2,
        **common_options,
    )
    model.insert(
        dense_indices[2] + 1,
        ClassRoutedLogicDense(
            in_dim=60_000,
            out_dim=30_000,
            num_classes=100,
            shared_input=False,
            topology_seed=topology_seed + 3,
            **common_options,
        ),
    )
    model[-1] = GroupSum(k=100, tau=10)
    return model.to(device)


def build_model_cifar100_small_classrouted_channelmixed(
    device: torch.device,
    *,
    parametrization: str,
    input_thresholds: torch.Tensor | None = None,
    input_encoding: str | None = None,
    num_quantile_bits: int = 2,
    num_sine_bits: int = 2,
    sine_sigma: float = 1.0,
    sine_seed: int = 0,
) -> torch.nn.Module:
    """CIFAR-100 class-routed model with balanced Conv1 channel mixing."""

    return build_model_cifar100_small_classrouted(
        device,
        parametrization=parametrization,
        input_thresholds=input_thresholds,
        input_encoding=input_encoding,
        num_quantile_bits=num_quantile_bits,
        num_sine_bits=num_sine_bits,
        sine_sigma=sine_sigma,
        sine_seed=sine_seed,
        first_conv_init_method="channel-balanced",
    )


def build_model_cifar10_deepconv(
    device: torch.device,
    *,
    parametrization: str,
    input_thresholds: torch.Tensor | None = None,
    input_encoding: str | None = None,
    num_quantile_bits: int = 2,
    num_sine_bits: int = 2,
    sine_sigma: float = 1.0,
    sine_seed: int = 0,
) -> torch.nn.Module:
    """Build CLGN-small with one additional sequential convolution."""

    encoder = _build_cifar10_encoder(
        input_thresholds,
        input_encoding=input_encoding,
        num_quantile_bits=num_quantile_bits,
        num_sine_bits=num_sine_bits,
        sine_sigma=sine_sigma,
        sine_seed=sine_seed,
    )
    encoded_model_type = type(
        "EncodedClgnCifar10DeepConv",
        (ClgnCifar10_deepconv,),
        {
            "n_input_bits": encoder.num_bits,
            "k_num": 36,
            "group_size_input": 1,
        },
    )
    model = encoded_model_type(
        thresholds=encoder.thresholds.new_empty((3, encoder.num_bits)),
        binarization="dummy",
        binarization_kwargs={},
        connections_kwargs={"init_method": "random-unique"},
        lut_rank=2,
        parametrization=parametrization,
        parametrization_kwargs={
            "forward_sampling": "soft",
            "temperature": 1.0,
        },
        device=str(device),
    )
    model[0] = encoder
    return model.to(device)

#under training
def build_model_cifar10(
    device: torch.device,
    *,
    parametrization: str,
    input_thresholds: torch.Tensor | None = None,
    input_encoding: str | None = None,
    num_quantile_bits: int = 2,
    num_sine_bits: int = 2,
    sine_sigma: float = 1.0,
    sine_seed: int = 0,
) -> torch.nn.Module:

    width = 4_000
    encoder = _build_cifar10_encoder(
        input_thresholds,
        input_encoding=input_encoding,
        num_quantile_bits=num_quantile_bits,
        num_sine_bits=num_sine_bits,
        sine_sigma=sine_sigma,
        sine_seed=sine_seed,
    )

    logic_options = {
        "lut_rank": 4,
        "parametrization": parametrization,
        "parametrization_kwargs": {
            "forward_sampling": "soft",
            "temperature": 1.0,
        },
        "device": str(device),
    }

    return torch.nn.Sequential(

        encoder,

        # --------------------------------------------------
        # Stage 1
        # (3 * num_input_bits) x 32 x 32 -> 32 x 30 x 30
        # --------------------------------------------------
        LogicConv2d(
            in_dim=32,
            channels=encoder.output_channels,
            num_kernels=32,
            tree_depth=2,
            receptive_field_size=3,
            **logic_options,
        ),

        # -> 32 x 15 x 15
        OrPooling2d(
            kernel_size=2,
            stride=2,
            padding=0,
        ),

        # --------------------------------------------------
        # Stage 2
        # -> 64 x 13 x 13
        # --------------------------------------------------
        LogicConv2d(
            in_dim=15,
            channels=32,
            num_kernels=64,
            tree_depth=2,
            receptive_field_size=3,
            **logic_options,
        ),

        # -> 64 x 6 x 6
        OrPooling2d(
            kernel_size=2,
            stride=2,
            padding=0,
        ),

        # --------------------------------------------------
        # Same-shape Boolean residual refinement
        #
        # 64 x 6 x 6 -> 64 x 6 x 6
        # The residual branch starts closed, so discrete initialization is an
        # exact identity and does not immediately saturate the OR skip.
        # --------------------------------------------------
        CifarResidualLogicBlock(
            channels=64,
            spatial_dim=6,
            tree_depth=2,
            **logic_options,
        ),

        # --------------------------------------------------
        # Stage 3
        # -> 128 x 4 x 4
        # --------------------------------------------------
        LogicConv2d(
            in_dim=6,
            channels=64,
            num_kernels=128,
            tree_depth=2,
            receptive_field_size=3,
            **logic_options,
        ),

        # -> 128 x 2 x 2
        OrPooling2d(
            kernel_size=2,
            stride=2,
            padding=0,
        ),

        torch.nn.Flatten(),

        # --------------------------------------------------
        # Classifier
        #
        # 512 -> 2000 -> GroupSum
        #
        # Removed the second 2000 -> 2000 LogicDense
        # to reduce memorization capacity.
        # --------------------------------------------------
        LogicDense(
            128 * 2 * 2,
            width,
            **logic_options,
        ),

        GroupSum(
            k=10,
            tau=8,
        ),

    ).to(device)



def build_model_cifar10_linear_head(
    device: torch.device,
    *,
    parametrization: str,
    input_thresholds: torch.Tensor | None = None,
    input_encoding: str | None = None,
    num_quantile_bits: int = 2,
    num_sine_bits: int = 2,
    sine_sigma: float = 1.0,
    sine_seed: int = 0,
) -> torch.nn.Module:
    """Build the linear-head architecture for encoded RGB CIFAR-10."""

    width = 16_000
    num_kernels = 64
    encoder = _build_cifar10_encoder(
        input_thresholds,
        input_encoding=input_encoding,
        num_quantile_bits=num_quantile_bits,
        num_sine_bits=num_sine_bits,
        sine_sigma=sine_sigma,
        sine_seed=sine_seed,
    )

    logic_options = {
        "lut_rank": 4,
        "parametrization": parametrization,
        "parametrization_kwargs": {
            "forward_sampling": "soft",
            "temperature": 1.0,
        },
        "device": str(device),
    }

    return torch.nn.Sequential(
        encoder,
        LogicConv2d(
            in_dim=32,
            channels=encoder.output_channels,
            num_kernels=num_kernels,
            tree_depth=2,
            receptive_field_size=3,
            **logic_options,
        ),
        OrPooling2d(kernel_size=2, stride=2, padding=0),
        torch.nn.Flatten(),
        LogicDense(num_kernels * 15 * 15, width, **logic_options),
        LogicDense(width, width, **logic_options),
        torch.nn.Linear(width, 10),
    ).to(device)


MODEL_BUILDERS = {
    "build_model": ("mnist", build_model),
    "build_model_linear_head": ("mnist", build_model_linear_head),
    "build_model_cifar10": ("cifar10", build_model_cifar10),
    "build_model_cifar10_org": ("cifar10", build_model_cifar10_org),
    "build_model_cifar10_smallmodel": (
        "cifar10",
        build_model_cifar10_smallmodel,
    ),
    "build_model_cifar10_small_modifieddense": (
        "cifar10",
        build_model_cifar10_small_modifieddense,
    ),
    "build_model_cifar10_small_classrouted": (
        "cifar10",
        build_model_cifar10_small_classrouted,
    ),
    "build_model_cifar10_small_classrouted_channelmixed": (
        "cifar10",
        build_model_cifar10_small_classrouted_channelmixed,
    ),
    "build_model_cifar100_small_classrouted": (
        "cifar100",
        build_model_cifar100_small_classrouted,
    ),
    "build_model_cifar100_small_classrouted_channelmixed": (
        "cifar100",
        build_model_cifar100_small_classrouted_channelmixed,
    ),
    "build_model_cifar10_deepconv": (
        "cifar10",
        build_model_cifar10_deepconv,
    ),
    "build_model_cifar10_linear_head": (
        "cifar10",
        build_model_cifar10_linear_head,
    ),
}


def build_model_for_dataset(
    device: torch.device,
    *,
    dataset: str,
    model_variant: str,
    parametrization: str,
    input_thresholds: torch.Tensor | None = None,
    input_encoding: str | None = None,
    num_quantile_bits: int = 2,
    num_sine_bits: int = 2,
    sine_sigma: float = 1.0,
    sine_seed: int = 0,
) -> torch.nn.Module:
    """Build the requested dataset and classification-head combination."""

    try:
        builder_dataset, builder = MODEL_BUILDERS[model_variant]
    except KeyError as error:
        raise ValueError(
            f"Unknown model builder {model_variant!r}; "
            f"choose from {tuple(MODEL_BUILDERS)}."
        ) from error
    if builder_dataset != dataset:
        raise ValueError(
            f"Model builder {model_variant!r} is for {builder_dataset}, "
            f"not {dataset}."
        )
    builder_kwargs = {"parametrization": parametrization}
    if dataset in {"cifar10", "cifar100"}:
        builder_kwargs.update(
            {
                "input_thresholds": input_thresholds,
                "input_encoding": input_encoding,
                "num_quantile_bits": num_quantile_bits,
                "num_sine_bits": num_sine_bits,
                "sine_sigma": sine_sigma,
                "sine_seed": sine_seed,
            }
        )
    return builder(device, **builder_kwargs)


def logic_layer_width_report(
    model: torch.nn.Module,
) -> list[dict[str, int | str]]:
    """Describe output-wire and LUT-node widths for every logic layer."""

    report = []
    for name, module in model.named_modules():
        if isinstance(module, LogicConv2d):
            luts_per_output = sum(
                module.lut_rank**level
                for level in range(module.tree_depth)
            )
            output_wires = module.num_kernels * module.n_kernel_positions
            report.append(
                {
                    "layer": name,
                    "type": type(module).__name__,
                    "width": output_wires,
                    "output_wires": output_wires,
                    "shared_lut_nodes": module.num_kernels * luts_per_output,
                    "lut_node_applications": output_wires * luts_per_output,
                }
            )
        elif isinstance(module, LogicDense):
            report.append(
                {
                    "layer": name,
                    "type": type(module).__name__,
                    "width": module.out_dim,
                    "output_wires": module.out_dim,
                    "shared_lut_nodes": module.out_dim,
                    "lut_node_applications": module.out_dim,
                }
            )
    return report


## attention block for CIFAR-10 model    
## seems not to improve accuracy, but we keep it for future experiments

class LogicChannelAttention(torch.nn.Module):
    """
    Identity-initialized Boolean channel excitation.

    For input [B, 64, 6, 6]:

        spatial descriptor:
            36 -> 9 -> 4 -> 1 per channel

        local channel gate:
            each gate sees the descriptors of its two neighbours

        output:
            x OR (gate AND neighbouring_features)

    The final gate is initialized to constant zero, so the block starts as an
    exact identity at discrete inference.  Its relaxed form is a probabilistic
    Boolean expression in [0, 1], preserving an unattenuated gradient path
    through ``x`` while the attention branch starts learning.
    """

    def __init__(
        self,
        channels: int,
        height: int,
        width: int,
        *,
        channel_lut_rank: int = 2,
        **logic_options,
    ):
        super().__init__()

        self.channels = channels
        self.height = height
        self.width = width

        spatial_lut_rank = logic_options["lut_rank"]

        if spatial_lut_rank != 4:
            raise ValueError(
                "The spatial attention reduction currently expects lut_rank=4."
            )
        if channel_lut_rank != 2:
            raise ValueError("Local channel attention currently expects lut_rank=2.")

        # --------------------------------------------------
        # Learned spatial descriptor
        #
        # Avoid OR pooling here: max/OR over active feature maps rapidly
        # saturates and gives the channel gate nearly constant descriptors.
        # For the current 6x6 input, the balanced rank-4 reduction is
        # 36 -> 9 -> 4 -> 1.
        # --------------------------------------------------
        spatial_size = height * width
        if spatial_size != 36:
            raise ValueError(
                f"Expected a 6x6 feature map (36 values), got {spatial_size}."
            )

        self.spatial_summary = torch.nn.Sequential(
            LogicDense(
                36,
                9,
                **logic_options,
            ),
            LogicDense(
                9,
                4,
                **logic_options,
            ),
            LogicDense(4, 1, **logic_options),
        )

        # --------------------------------------------------
        # Shallow, local channel interaction.
        # Each output channel sees only its immediate left/right neighbours.
        # This avoids the destructive 64 -> 16 -> 4 -> 16 -> 64 bottleneck.
        # --------------------------------------------------
        channel_options = {
            **logic_options,
            "lut_rank": channel_lut_rank,
        }
        self.channel_gate = LogicDense(
            channels,
            channels,
            **channel_options,
        )

        channel_ids = torch.arange(channels, device=self.channel_gate.weight.device)
        local_indices = torch.stack(
            ((channel_ids - 1) % channels, (channel_ids + 1) % channels)
        )
        with torch.no_grad():
            self.channel_gate.connections.indices.copy_(local_indices)
            self._initialize_closed_gate(
                parametrization=logic_options["parametrization"]
            )

        self.last_gate = None

    def _initialize_closed_gate(self, *, parametrization: str) -> None:
        """Initialize the discrete attention branch to constant false."""

        weight = self.channel_gate.weight
        if parametrization == "light":
            # Every truth-table entry is negative -> constant-zero LUT.
            weight.fill_(-3.0)
        elif parametrization == "warp":
            # A positive constant Walsh coefficient produces sigmoid(-c)
            # during relaxed training and a constant-zero discrete LUT.
            weight.zero_()
            weight[:, 0] = 3.0
        else:
            raise ValueError(
                "LogicChannelAttention supports 'light' and 'warp' "
                f"parametrizations, got {parametrization!r}."
            )

    def forward(self, x):
        B, C, H, W = x.shape

        if C != self.channels:
            raise ValueError(
                f"Expected {self.channels} channels, got {C}"
            )

        if H != self.height or W != self.width:
            raise ValueError(
                f"Expected {self.height}x{self.width}, got {H}x{W}"
            )

        # --------------------------------------------------
        # Spatial descriptor
        # --------------------------------------------------

        # [B,C,6,6] -> [B*C,36]
        z = x.reshape(B * C, H * W)

        # 36 -> 9 -> 4 -> 1
        z = self.spatial_summary(z)

        # [B*C,1] -> [B,C]
        z = z.reshape(B, C)

        # --------------------------------------------------
        # Local channel interaction: each gate reads the descriptors of the
        # adjacent channels selected by ``local_indices`` above.
        # --------------------------------------------------
        gate = self.channel_gate(z)

        gate = gate.reshape(B, C, 1, 1)

        # Keep for debugging/analysis
        self.last_gate = gate.detach()

        # Neighbouring evidence is a probabilistic OR.  The output is then
        # x OR (gate AND neighbour), which is exactly Boolean for binary
        # inputs/gates and remains in [0, 1] in relaxed mode.
        left = torch.roll(x, shifts=1, dims=1)
        right = torch.roll(x, shifts=-1, dims=1)
        neighbour = left + right - left * right
        excitation = gate * neighbour
        return x + excitation - x * excitation



class CifarResidualLogicBlock(torch.nn.Module):
    """Same-shape residual block for the rank-4 CIFAR logic model.

    The main branch contains two logic convolutions.  Its final root LUT is
    initialized to constant false, making the discrete block an exact identity
    at initialization.  The skip merge uses relaxed OR during training and is
    exactly Boolean when its operands are binary.
    """

    def __init__(
        self,
        channels: int,
        spatial_dim: int,
        *,
        tree_depth: int = 2,
        receptive_field_size: int = 3,
        close_strength: float = 5.0,
        **logic_options,
    ):
        super().__init__()

        if logic_options["lut_rank"] != 4:
            raise ValueError("CifarResidualLogicBlock currently expects lut_rank=4.")
        if tree_depth < 1:
            raise ValueError("tree_depth must be at least 1.")

        padding = receptive_field_size // 2
        self.channels = channels
        self.spatial_dim = spatial_dim

        self.main = torch.nn.Sequential(
            LogicConv2d(
                in_dim=spatial_dim,
                channels=channels,
                num_kernels=channels,
                tree_depth=tree_depth,
                receptive_field_size=receptive_field_size,
                padding=padding,
                **logic_options,
            ),
            LogicConv2d(
                in_dim=spatial_dim,
                channels=channels,
                num_kernels=channels,
                tree_depth=tree_depth,
                receptive_field_size=receptive_field_size,
                padding=padding,
                **logic_options,
            ),
        )

        with torch.no_grad():
            self._initialize_closed_branch(
                parametrization=logic_options["parametrization"],
                close_strength=close_strength,
            )

        self.last_branch = None

    def _initialize_closed_branch(
        self,
        *,
        parametrization: str,
        close_strength: float,
    ) -> None:
        """Make the final root LUT discrete-zero with a small asymmetry."""

        final_conv = self.main[-1]
        root_weight = final_conv.tree_weights[-1]
        noise = 0.01 * torch.randn_like(root_weight)

        if parametrization == "light":
            # All truth-table coefficients remain negative, hence the root is
            # a constant-zero LUT after discretization.
            root_weight.fill_(-close_strength)
            root_weight.add_(noise)
        elif parametrization == "warp":
            # A dominant positive constant Walsh coefficient keeps every
            # reconstructed truth-table value positive -> discrete zero.
            root_weight.zero_()
            root_weight[..., 0] = close_strength
            root_weight.add_(noise)
        else:
            raise ValueError(
                "CifarResidualLogicBlock supports 'light' and 'warp' "
                f"parametrizations, got {parametrization!r}."
            )

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError(f"Expected a 4D tensor, got shape {tuple(x.shape)}.")
        if x.shape[1:] != (
            self.channels,
            self.spatial_dim,
            self.spatial_dim,
        ):
            raise ValueError(
                "Expected input shape "
                f"[B,{self.channels},{self.spatial_dim},{self.spatial_dim}], "
                f"got {tuple(x.shape)}."
            )

        branch = self.main(x)
        self.last_branch = branch.detach()

        # Probabilistic OR: bounded in [0, 1] for relaxed operands and exactly
        # OR for discrete operands.
        return x + branch - x * branch
