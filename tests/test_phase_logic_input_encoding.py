import numpy as np
import pytest
import torch

from torchlogix.layers import (
    ClassRoutedLogicDense,
    GroupSum,
    LogicConv2d,
    LogicDense,
)

from examples.phase_logic.config import ExperimentConfig
from examples.phase_logic.data import compute_cifar10_quantile_thresholds
from examples.phase_logic.model import (
    HybridFeatureEncoder,
    QuantileThermometerEncoder,
    StraightThroughBinarize,
    build_model_cifar10,
    build_model_cifar10_org,
    build_model_cifar10_deepconv,
    build_model_cifar10_small_classrouted,
    build_model_cifar10_small_classrouted_channelmixed,
    build_model_cifar10_smallmodel,
    build_model_cifar100_small_classrouted,
    build_model_cifar100_small_classrouted_channelmixed,
    logic_layer_width_report,
)
from examples.phase_logic.schedules import cosine_decay


class _FakeCifar10:
    def __init__(self, *, train: bool = True):
        self.train = train
        self.data = np.array(
            [
                [[[0, 5, 2]]],
                [[[10, 15, 12]]],
                [[[20, 25, 22]]],
                [[[30, 35, 32]]],
                [[[40, 45, 42]]],
                [[[50, 55, 52]]],
            ],
            dtype=np.uint8,
        )


def test_quantile_thresholds_are_channelwise_training_order_statistics():
    thresholds = compute_cifar10_quantile_thresholds(
        _FakeCifar10(),
        num_bits=2,
    )

    expected = torch.tensor(
        [
            [20 / 255, 40 / 255],
            [25 / 255, 45 / 255],
            [22 / 255, 42 / 255],
        ],
        dtype=torch.float32,
    )
    torch.testing.assert_close(thresholds, expected)


def test_quantile_thresholds_reject_test_split():
    with pytest.raises(ValueError, match="training set only"):
        compute_cifar10_quantile_thresholds(
            _FakeCifar10(train=False),
            num_bits=2,
        )


def test_thermometer_encoder_channel_order_shape_and_monotonic_codes():
    encoder = QuantileThermometerEncoder(
        torch.tensor(
            [
                [0.25, 0.75],
                [0.25, 0.75],
                [0.25, 0.75],
            ]
        )
    )
    image = torch.tensor([[[[0.10]], [[0.50]], [[0.90]]]])

    encoded = encoder(image)

    assert encoded.shape == (1, 6, 1, 1)
    torch.testing.assert_close(
        encoded.flatten(),
        torch.tensor([0.0, 0.0, 1.0, 0.0, 1.0, 1.0]),
    )
    assert "thresholds" in encoder.state_dict()
    assert list(encoder.parameters()) == []


def test_straight_through_binarizer_has_hard_forward_and_identity_backward():
    values = torch.tensor([-1.0, 0.0, 2.0], requires_grad=True)

    encoded = StraightThroughBinarize.apply(values)
    encoded.sum().backward()

    torch.testing.assert_close(encoded, torch.tensor([0.0, 0.0, 1.0]))
    torch.testing.assert_close(values.grad, torch.ones(3))


def test_sinusoidal_encoder_is_seeded_binary_and_parameter_free():
    rng_state = torch.random.get_rng_state()
    first = HybridFeatureEncoder(
        mode="sinusoidal",
        num_sine_bits=4,
        sigma=1.0,
        sine_seed=17,
    )
    second = HybridFeatureEncoder(
        mode="sinusoidal",
        num_sine_bits=4,
        sigma=1.0,
        sine_seed=17,
    )

    assert torch.equal(torch.random.get_rng_state(), rng_state)
    torch.testing.assert_close(first.sine_weights, second.sine_weights)
    torch.testing.assert_close(first.sine_phases, second.sine_phases)
    encoded = first(torch.rand(2, 3, 4, 4))
    assert encoded.shape == (2, 12, 4, 4)
    assert set(encoded.unique().tolist()).issubset({0.0, 1.0})
    assert {"thresholds", "sine_weights", "sine_phases"}.issubset(
        first.state_dict()
    )
    assert list(first.parameters()) == []


def test_hybrid_encoder_orders_quantile_then_sine_bits_per_rgb_channel():
    encoder = HybridFeatureEncoder(
        mode="hybrid",
        thresholds=torch.tensor([[0.25, 0.75]] * 3),
        num_quantile_bits=2,
        num_sine_bits=2,
        sine_seed=3,
    )
    encoder.sine_weights.zero_()
    encoder.sine_phases.copy_(
        torch.tensor([[torch.pi / 2, -torch.pi / 2]] * 3)
    )
    image = torch.tensor([[[[0.10]], [[0.50]], [[0.90]]]])

    encoded = encoder(image)

    assert encoded.shape == (1, 12, 1, 1)
    torch.testing.assert_close(
        encoded.flatten(),
        torch.tensor(
            [
                0.0, 0.0, 1.0, 0.0,
                1.0, 0.0, 1.0, 0.0,
                1.0, 1.0, 1.0, 0.0,
            ]
        ),
    )


def test_cifar_builder_uses_encoder_output_channel_count():
    thresholds = torch.tensor(
        [
            [0.2, 0.4, 0.6, 0.8],
            [0.2, 0.4, 0.6, 0.8],
            [0.2, 0.4, 0.6, 0.8],
        ]
    )

    model = build_model_cifar10(
        torch.device("cpu"),
        parametrization="light",
        input_thresholds=thresholds,
    )

    assert isinstance(model[0], QuantileThermometerEncoder)
    assert model[0].output_channels == 12
    assert model[1].channels == 12
    assert "0.thresholds" in model.state_dict()


def test_original_cifar_builder_uses_encoder_output_channel_count():
    thresholds = torch.tensor(
        [
            [0.2, 0.4],
            [0.2, 0.4],
            [0.2, 0.4],
        ]
    )

    model = build_model_cifar10_org(
        torch.device("cpu"),
        parametrization="light",
        input_thresholds=thresholds,
    )

    assert isinstance(model[0], QuantileThermometerEncoder)
    assert model[0].output_channels == 6
    assert model[1].channels == 6
    assert "0.thresholds" in model.state_dict()


def test_clgn_small_builder_uses_external_encoder_output_channel_count():
    thresholds = torch.tensor(
        [
            [0.2, 0.4, 0.6, 0.8],
            [0.2, 0.4, 0.6, 0.8],
            [0.2, 0.4, 0.6, 0.8],
        ]
    )

    model = build_model_cifar10_smallmodel(
        torch.device("cpu"),
        parametrization="light",
        input_thresholds=thresholds,
    )

    assert isinstance(model[0], QuantileThermometerEncoder)
    assert model[0].output_channels == 12
    assert model[1].channels == 12
    assert "0.thresholds" in model.state_dict()
    assert all(
        module.connections.init_method == "random-unique"
        for module in model
        if isinstance(module, LogicConv2d)
    )


def test_clgn_small_builder_accepts_equal_budget_hybrid_encoder():
    thresholds = torch.tensor([[0.33, 0.66]] * 3)

    model = build_model_cifar10_smallmodel(
        torch.device("cpu"),
        parametrization="light",
        input_thresholds=thresholds,
        input_encoding="hybrid",
        num_quantile_bits=2,
        num_sine_bits=2,
        sine_sigma=1.0,
        sine_seed=9,
    )

    assert isinstance(model[0], HybridFeatureEncoder)
    assert model[0].mode == "hybrid"
    assert model[0].output_channels == 12
    assert model[1].channels == 12
    assert "0.sine_weights" in model.state_dict()


def test_cifar100_uses_shared_dense1_and_late_class_routing():
    model = build_model_cifar100_small_classrouted(
        torch.device("cpu"),
        parametrization="light",
        input_thresholds=torch.tensor([[0.33, 0.66]] * 3),
        input_encoding="quantile",
        num_quantile_bits=2,
    )

    dense = [module for module in model if isinstance(module, LogicDense)]
    assert [module.out_dim for module in dense] == [
        60_000,
        60_000,
        60_000,
        30_000,
    ]
    assert all(type(module) is LogicDense for module in dense[:2])
    assert all(isinstance(module, ClassRoutedLogicDense) for module in dense[2:])
    assert all(module.num_classes == 100 for module in dense[2:])
    assert dense[2].shared_input is True
    assert dense[2].shared_input_subset is True
    assert dense[3].shared_input is False
    assert dense[3].routed_input_reuse is False
    assert dense[2].connections.indices.max() >= 600
    assert all(module.lut_rank == 2 for module in dense)
    assert isinstance(model[-1], GroupSum)
    assert model[-1].k == 100
    assert model[-1].tau == 10
    assert [entry["width"] for entry in logic_layer_width_report(model)][-4:] == [
        60_000,
        60_000,
        60_000,
        30_000,
    ]

    model.eval()
    assert model(torch.rand(1, 3, 32, 32)).shape == (1, 100)


def test_cifar100_configuration_accepts_only_its_model_variant():
    config = ExperimentConfig(
        dataset="cifar100",
        model_variant="build_model_cifar100_small_classrouted",
        input_encoding="quantile",
        num_input_bits=2,
    )

    config.validate()
    assert "cifar100" in str(config.run_dir)


@pytest.mark.parametrize(
    ("baseline_builder", "mixed_builder", "num_classes"),
    [
        (
            build_model_cifar10_small_classrouted,
            build_model_cifar10_small_classrouted_channelmixed,
            10,
        ),
        (
            build_model_cifar100_small_classrouted,
            build_model_cifar100_small_classrouted_channelmixed,
            100,
        ),
    ],
)
def test_channel_mixed_models_change_only_first_conv(
    baseline_builder,
    mixed_builder,
    num_classes,
):
    kwargs = {
        "device": torch.device("cpu"),
        "parametrization": "light",
        "input_thresholds": torch.tensor([[0.33, 0.66]] * 3),
        "input_encoding": "quantile",
        "num_quantile_bits": 2,
    }
    torch.manual_seed(7)
    baseline = baseline_builder(**kwargs)
    torch.manual_seed(7)
    mixed = mixed_builder(**kwargs)

    baseline_convs = [m for m in baseline if isinstance(m, LogicConv2d)]
    mixed_convs = [m for m in mixed if isinstance(m, LogicConv2d)]
    assert [m.connections.init_method for m in baseline_convs] == [
        "random-unique"
    ] * 4
    assert [m.connections.init_method for m in mixed_convs] == [
        "channel-balanced",
        "random-unique",
        "random-unique",
        "random-unique",
    ]
    assert [m.connections.channel_group_size for m in baseline_convs] == [
        1,
        2,
        2,
        2,
    ]
    assert [m.connections.channel_group_size for m in mixed_convs[1:]] == [
        2,
        2,
        2,
    ]
    assert sum(p.numel() for p in mixed.parameters()) == sum(
        p.numel() for p in baseline.parameters()
    )

    mixed.eval()
    assert mixed(torch.rand(1, 3, 32, 32)).shape == (1, num_classes)


@pytest.mark.parametrize(
    ("encoding", "quantile_bits", "sine_bits"),
    [
        ("hybrid", 2, 2),
        ("quantile", 4, 0),
        ("hybrid", 4, 4),
        ("quantile", 8, 0),
    ],
)
def test_channel_mixed_conv_tracks_dynamic_encoder_width(
    encoding,
    quantile_bits,
    sine_bits,
):
    thresholds = torch.linspace(0.1, 0.9, quantile_bits).repeat(3, 1)
    model = build_model_cifar10_small_classrouted_channelmixed(
        torch.device("cpu"),
        parametrization="light",
        input_thresholds=thresholds,
        input_encoding=encoding,
        num_quantile_bits=quantile_bits,
        num_sine_bits=sine_bits,
    )
    first_conv = next(m for m in model if isinstance(m, LogicConv2d))
    expected_channels = 3 * (quantile_bits + sine_bits)
    assert first_conv.channels == expected_channels

    first_position = first_conv.connections.indices[0][:, :, 0, :, -1]
    per_kernel = first_position.permute(1, 0, 2).reshape(
        first_conv.num_kernels,
        -1,
    )
    assert all(row.unique().numel() == row.numel() for row in per_kernel)
    usage = torch.bincount(per_kernel.flatten(), minlength=expected_channels)
    assert int(usage.max() - usage.min()) <= 1


def test_clgn_deepconv_adds_one_shape_preserving_convolution():
    thresholds = torch.tensor([[0.33, 0.66]] * 3)
    small = build_model_cifar10_smallmodel(
        torch.device("cpu"),
        parametrization="light",
        input_thresholds=thresholds,
        input_encoding="hybrid",
        num_quantile_bits=2,
        num_sine_bits=2,
    )
    deep = build_model_cifar10_deepconv(
        torch.device("cpu"),
        parametrization="light",
        input_thresholds=thresholds,
        input_encoding="hybrid",
        num_quantile_bits=2,
        num_sine_bits=2,
    )

    small_convs = [module for module in small if isinstance(module, LogicConv2d)]
    deep_convs = [module for module in deep if isinstance(module, LogicConv2d)]
    small_dense = next(module for module in small if isinstance(module, LogicDense))
    deep_dense = next(module for module in deep if isinstance(module, LogicDense))

    assert len(small_convs) == 4
    assert len(deep_convs) == 5
    assert all(
        module.connections.init_method == "random-unique"
        for module in deep_convs
    )
    assert deep_convs[-1].in_dim == (4, 4)
    assert deep_convs[-1].channels == deep_convs[-2].num_kernels
    assert deep_convs[-1].num_kernels == deep_convs[-2].num_kernels
    assert small_dense.in_dim == 128 * 36
    assert deep_dense.in_dim == 128 * 36

    deep.eval()
    assert deep(torch.rand(1, 3, 32, 32)).shape == (1, 10)


@pytest.mark.parametrize(
    ("encoding", "bits"),
    [
        ("fixed", 1),
        ("quantile", 2),
        ("quantile", 4),
        ("sinusoidal", 4),
    ],
)
def test_supported_cifar_encoding_configurations(encoding, bits):
    config = ExperimentConfig(
        dataset="cifar10",
        model_variant="build_model_cifar10",
        input_encoding=encoding,
        num_input_bits=bits,
    )

    config.validate()
    assert config.input_encoding_slug in str(config.run_dir)


def test_supported_equal_budget_hybrid_configuration():
    config = ExperimentConfig(
        dataset="cifar10",
        model_variant="build_model_cifar10_smallmodel",
        input_encoding="hybrid",
        num_quantile_bits=2,
        num_sine_bits=2,
        sine_sigma=1.0,
        sine_seed=7,
    )

    config.validate()
    assert config.total_input_bits == 4
    assert "hybrid_q2_s2_sigma_1_seed_7" in str(config.run_dir)


def test_cifar_augmentation_uses_a_separate_run_directory():
    config = ExperimentConfig(
        dataset="cifar10",
        model_variant="build_model_cifar10_smallmodel",
        input_encoding="hybrid",
        num_quantile_bits=2,
        num_sine_bits=2,
        cifar_augmentation=True,
    )

    config.validate()
    assert "crop_flip_augmentation" in str(config.run_dir)


def test_cifar_augmentation_is_rejected_for_mnist():
    config = ExperimentConfig(cifar_augmentation=True)

    with pytest.raises(ValueError, match="CIFAR augmentation"):
        config.validate()


def test_cosine_learning_rate_decay_hits_both_endpoints():
    assert cosine_decay(1, 57_000, 0.02, 0.002) == pytest.approx(0.02)
    assert cosine_decay(57_000, 57_000, 0.02, 0.002) == pytest.approx(0.002)


def test_cosine_learning_rate_configuration():
    config = ExperimentConfig(
        lr_schedule="cosine",
        end_learning_rate=0.002,
    )

    config.validate()
    assert config.end_temperature == pytest.approx(0.30)
    assert "cosine_lr_0p02_to_0p002" in str(config.run_dir)


@pytest.mark.parametrize(
    ("encoding", "bits"),
    [("fixed", 2), ("fixed", 4), ("quantile", 1)],
)
def test_invalid_cifar_encoding_configurations(encoding, bits):
    config = ExperimentConfig(
        dataset="cifar10",
        model_variant="build_model_cifar10",
        input_encoding=encoding,
        num_input_bits=bits,
    )

    with pytest.raises(ValueError):
        config.validate()
