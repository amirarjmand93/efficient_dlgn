import torch

from torchlogix.layers import LogicConv2d
from torchlogix.utils import set_export_mode


def _build_conv(*, stride=1):
    return LogicConv2d(
        in_dim=8,
        channels=3,
        num_kernels=4,
        tree_depth=2,
        receptive_field_size=3,
        stride=stride,
        padding=1,
        parametrization="light",
        parametrization_kwargs={
            "forward_sampling": "soft",
            "temperature": 1.0,
        },
    )


def test_stride_two_halves_spatial_resolution_in_relaxed_and_discrete_modes():
    layer = _build_conv(stride=2)
    relaxed_input = torch.rand(2, 3, 8, 8)

    layer.train()
    relaxed = layer(relaxed_input)
    assert relaxed.shape == (2, 4, 4, 4)

    layer.eval()
    discrete_input = torch.randint(0, 2, (2, 3, 8, 8)).float()
    discrete = layer(discrete_input)
    assert discrete.shape == (2, 4, 4, 4)
    assert torch.all((discrete == 0) | (discrete == 1))


def test_stride_two_export_matches_discrete_forward():
    layer = _build_conv(stride=2)
    values = torch.randint(0, 2, (1, 3, 8, 8), dtype=torch.bool)

    layer.eval()
    expected = layer(values.float())
    set_export_mode(layer)
    exported = layer(values)

    assert exported.shape == (1, 4, 4, 4)
    assert torch.equal(exported, expected.bool())


def test_default_non_strided_shape_is_unchanged():
    layer = _build_conv()

    assert layer.stride == 1
    assert layer(torch.rand(2, 3, 8, 8)).shape == (2, 4, 8, 8)
