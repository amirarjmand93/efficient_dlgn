"""Focused MRLB topology, routing persistence, and CUDA parity checks."""

from dataclasses import replace

import pytest
import torch

from examples.phase_logic.compact import CompactLogicController
from examples.phase_logic.config import ExperimentConfig
from examples.phase_logic.hybrid_freezing import HybridFreezeController
from examples.phase_logic.model import build_model_for_dataset, logic_layer_width_report
from examples.phase_logic.mrlb_compact import RoutedSpatialCandidates
from examples.phase_logic.run_experiment import build_parser, config_from_args


@pytest.fixture(autouse=True, scope="module")
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def make_model(variant="build_model_cifar10_small_mrlb_compact_head", **kwargs):
    return build_model_for_dataset(
        torch.device(kwargs.pop("device", "cpu")), dataset="cifar10", model_variant=variant,
        parametrization="compact", input_encoding="hybrid", hybrid_features="lbp",
        input_thresholds=torch.tensor([[0.3, 0.7]]).expand(3, -1),
        compact_backend=kwargs.pop("backend", "torch"), **kwargs,
    )


def assert_bounds(layer):
    offsets = layer.impl.offsets_ch
    assert torch.all(offsets >= layer.channel_ranges[..., 0, None])
    assert torch.all(offsets < layer.channel_ranges[..., 1, None])
    for coordinates in (layer.impl.offsets_h, layer.impl.offsets_w):
        assert torch.all(coordinates.abs() <= layer.impl.padding)
    same = (offsets[0] == offsets[1])
    same &= layer.impl.offsets_h[0] == layer.impl.offsets_h[1]
    same &= layer.impl.offsets_w[0] == layer.impl.offsets_w[1]
    assert not same.any()


def test_registry_topology_and_unchanged_baseline_components():
    args = build_parser().parse_args([
        "--dataset", "cifar10", "--model-variant", "build_model_cifar10_small_mrlb_compact_head",
        "--parametrization", "compact", "--experiment", "pure_ce", "--iterations", "700000",
        "--input-encoding", "hybrid", "--hybrid-features", "lbp",
        "--mrlb-conv3-relational-channels", "128",
    ])
    config = config_from_args(args)
    config.validate()
    assert "mrlb_v2_local64r1_cross64r2_route128" in str(config.run_dir)
    with pytest.raises(ValueError, match="even"):
        replace(config, mrlb_conv3_relational_channels=127).validate()
    torch.manual_seed(4242449)
    baseline = make_model("build_model_cifar10_small_classrouted_compact_head")
    torch.manual_seed(4242449)
    model = make_model()
    for old, new in ((0, 0), (1, 1), (2, 2), (4, 6), (6, 8), (7, 9), (8, 10)):
        for key, value in baseline[old].state_dict().items():
            torch.testing.assert_close(value, model[new].state_dict()[key], rtol=0, atol=0)
    assert [model[i].out_dim for i in model.backbone_conv_indices] == [128, 128, 512, 512]
    assert model[3].out_dim == model[4].out_dim == 64
    assert model[3].impl.padding == 1 and model[4].impl.padding == 2
    assert model[5].in_shape == (384, 16, 16)
    assert model[8].impl.in_dim == 65536
    assert model[8].out_dim == model[9].out_dim == 88000
    assert model[-1].tau == baseline[-1].tau == 55
    assert model.relational_bridge_report()["conv3_output_routes"] == {
        "original": 384, "local_to_original": 64, "cross_to_original": 64,
    }
    for i in (3, 4, 5):
        assert_bounds(model[i])
    # The mixed Conv3 row domains must survive a full weak-row refresh too.
    with torch.no_grad():
        model[5].weights.zero_()
    model[5].resample(torch.arange(512), torch.optim.Adam(model.parameters()))
    assert_bounds(model[5])
    assert torch.all(model[4].impl.offsets_ch[0] < 128)
    assert torch.all(model[4].impl.offsets_ch[1] >= 128)
    assert sum(p.numel() for p in model.parameters()) == 2_838_528
    assert sum(r["lut_node_applications"] for r in logic_layer_width_report(model)) == 339_840
    controller = CompactLogicController(model, config)
    assert [name for name, _ in controller.convs] == ["1", "2", "3", "4", "5", "6"]
    hybrid = HybridFreezeController(model, replace(config, compact_freeze_mode="hybrid"))
    assert len(hybrid.convs) == 6
    # Verify both parallel branches and the unmodified bypass actually reach Conv3.
    captured = {}
    handles = [model[i].register_forward_hook(
        lambda module, inputs, output, i=i: captured.__setitem__(i, output)
    ) for i in (1, 2, 3, 4)]
    handles.append(model[5].register_forward_pre_hook(
        lambda module, inputs: captured.__setitem__("conv3_input", inputs[0])))
    try:
        with torch.no_grad():
            output = model.eval()(torch.rand(1, 3, 32, 32))
        assert output.shape == (1, 10) and torch.isfinite(output).all()
        torch.testing.assert_close(captured["conv3_input"],
                                   torch.cat([captured[i] for i in (1, 2, 3, 4)], dim=1))
    finally:
        for handle in handles:
            handle.remove()


@pytest.mark.parametrize("backend", ["torch", "cuda"])
@pytest.mark.parametrize("kernel_size", [3, 5])
def test_routing_resampling_winner_preservation_and_adam(backend, kernel_size):
    if backend == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    device = "cuda" if backend == "cuda" else "cpu"
    bounds = torch.tensor([[[0, 3]] * 4, [[3, 6]] * 4])
    layer = RoutedSpatialCandidates(
        device=device, backend=backend, in_dim=6 * 5 * 5, out_dim=4,
        in_shape=(6, 5, 5), kernel_size=kernel_size, channel_ranges=bounds,
    )
    optimizer = torch.optim.Adam(layer.parameters(), amsgrad=True)
    layer(torch.rand(2, 6, 5, 5, device=device)).sum().backward()
    optimizer.step()
    with torch.no_grad():
        layer.weights[0].zero_()
        layer.weights[0, 3] = 20
        layer.weights[1].zero_()
        for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            optimizer.state[layer.weights][key].fill_(1)
    before = {name: value.clone() for name, value in layer.impl.state_dict().items()}
    assert layer.resample(torch.tensor([0, 1], device=device), optimizer) == (1, 1)
    assert_bounds(layer)
    for key in ("gate_sequence", "offsets_ch", "offsets_h", "offsets_w"):
        if key.startswith("offsets"):
            torch.testing.assert_close(layer.impl.state_dict()[key][:, 0, 3], before[key][:, 0, 3])
            torch.testing.assert_close(layer.impl.state_dict()[key][:, 2:], before[key][:, 2:])
        else:
            torch.testing.assert_close(layer.impl.state_dict()[key][0, 3], before[key][0, 3])
            torch.testing.assert_close(layer.impl.state_dict()[key][2:], before[key][2:])
    torch.testing.assert_close(layer.weights[2:], before["weights"][2:])
    for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
        assert torch.count_nonzero(optimizer.state[layer.weights][key][:2]) == 0
        assert torch.all(optimizer.state[layer.weights][key][2:] == 1)
    layer.freeze()
    layer.train()
    assert torch.isfinite(layer(torch.rand(1, 6, 5, 5, device=device))).all()
    assert not layer.impl.training and not layer.weights.requires_grad


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_radius_two_cuda_matches_reference_and_full_model_backward():
    bounds = torch.tensor([[[0, 2]] * 4, [[2, 4]] * 4])
    common = dict(in_dim=4 * 5 * 5, out_dim=4, in_shape=(4, 5, 5),
                  kernel_size=5, channel_ranges=bounds)
    reference = RoutedSpatialCandidates(device="cpu", backend="torch", **common)
    actual = RoutedSpatialCandidates(device="cuda", backend="cuda", **common)
    actual.load_state_dict(reference.state_dict())
    x = torch.rand(2, 4, 5, 5, requires_grad=True)
    x_cuda = x.detach().cuda().requires_grad_()
    a, b = reference(x), actual(x_cuda)
    torch.testing.assert_close(a, b.cpu(), atol=2e-6, rtol=2e-5)
    a.square().sum().backward()
    b.square().sum().backward()
    torch.testing.assert_close(x.grad, x_cuda.grad.cpu(), atol=2e-5, rtol=2e-4)
    torch.testing.assert_close(reference.weights.grad, actual.weights.grad.cpu(), atol=2e-5, rtol=2e-4)
    torch.testing.assert_close(reference.eval()(x), actual.eval()(x_cuda).cpu(), atol=0, rtol=0)
    model = make_model(device="cuda", backend="cuda", mrlb_conv3_relational_channels=64)
    assert model.relational_bridge_report()["conv3_output_routes"] == {
        "original": 448, "local_to_original": 32, "cross_to_original": 32,
    }
    logits = model(torch.rand(2, 3, 32, 32, device="cuda"))
    torch.nn.functional.cross_entropy(logits, torch.tensor([1, 2], device="cuda")).backward()
    for i in (1, 2, 3, 4, 5, 6, 8, 9):
        assert model[i].weights.grad is not None
        assert torch.isfinite(model[i].weights.grad).all()
        assert torch.count_nonzero(model[i].weights.grad) > 0
