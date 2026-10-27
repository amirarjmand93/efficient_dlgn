"""Focused candidate-to-IR checks: truth tables, skips/padding, and existing C backend."""

import shutil

import numpy as np
import pytest
import torch

from torchlogix.circuit import Circuit
from torchlogix.layers import GroupSum
from examples.phase_logic.classrouted_compact import BitConcatCompactModel
from examples.phase_logic.compact import CandidateLogicLayer, UniformThermometerEncoder
from examples.phase_logic.dense_freezing import enable_dense_group_freezing
from examples.phase_logic.post_training.export_circuit import compact_to_circuit, verify_circuit


@pytest.fixture(autouse=True, scope="module")
def cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def test_all_operations_selected_by_slot_simplification_json_and_c(tmp_path):
    layer = CandidateLogicLayer(device="cpu", backend="torch", in_dim=12, out_dim=16)
    rows = torch.arange(16)
    slots = (rows + 5) % 16  # Winning slot differs from the operation ID.
    with torch.no_grad():
        layer.weights.zero_()
        layer.weights[rows, slots] = 20
        layer.impl.gate_sequence[rows, slots] = rows.to(torch.uint8)
        layer.impl.indices[0, rows, slots] = 0
        layer.impl.indices[1, rows, slots] = 1
    model = torch.nn.Sequential(UniformThermometerEncoder(1), torch.nn.Flatten(), layer,
                                GroupSum(k=4, tau=55)).eval()
    encoded = np.zeros((4, 3, 2, 2), dtype=bool)
    encoded.reshape(4, -1)[:, :2] = [[False, False], [False, True], [True, False], [True, True]]
    with torch.inference_mode():
        expected = (model(torch.from_numpy(encoded).float()) * 55).round().long().numpy()
    circuit = compact_to_circuit(model, input_shape=(3, 2, 2))
    verify_circuit(circuit, encoded, expected)
    circuit.simplify()
    verify_circuit(circuit, encoded, expected)
    # TRUE folding produces a bias even when variable reduction inputs remain.
    mixed = [sr for sr in circuit.sum_nodes if sr.beta > 0 and sr.input_ids]
    assert mixed
    rtl = circuit.get_verilog_code()
    for index, sr in enumerate(circuit.sum_nodes):
        if sr in mixed:
            assert f"s_{index} = {int(sr.beta)};" in rtl
    path = tmp_path / "circuit.json"
    circuit.write_json(str(path))
    verify_circuit(Circuit.from_json_file(str(path)), encoded, expected)
    if shutil.which("gcc"):
        circuit.compile(opt_level=0)
        verify_circuit(circuit, encoded, expected, compiled=True)


def test_skip_graph_spatial_offsets_zero_padding_and_frozen_dense_choices():
    torch.manual_seed(14)
    convs = []
    for shape, out, stride, kernel in [((3, 4, 4), 4, 2, 3), ((4, 2, 2), 4, 1, 5),
                                        ((8, 2, 2), 8, 2, 3), ((8, 1, 1), 8, 1, 3)]:
        convs.append(CandidateLogicLayer(device="cpu", backend="torch", in_dim=np.prod(shape),
                                        in_shape=shape, out_dim=out, stride=stride, kernel_size=kernel))
    dense = [CandidateLogicLayer(device="cpu", backend="torch", in_dim=16, out_dim=16) for _ in range(2)]
    model = BitConcatCompactModel(UniformThermometerEncoder(1), *convs, torch.nn.Flatten(),
                                  *dense, GroupSum(k=4, tau=7))
    with torch.no_grad():
        for layer in convs + dense:
            layer.weights.normal_()
    enable_dense_group_freezing(model)
    model[6].freeze_rows(model[6].proposal(torch.tensor([1, 3, 5])))
    model.eval()
    encoded = torch.randint(2, (5, 3, 4, 4)).bool().numpy()
    with torch.inference_mode():
        expected = (model(torch.from_numpy(encoded).float()) * 7).round().long().numpy()
    circuit = compact_to_circuit(model, input_shape=(3, 4, 4))
    verify_circuit(circuit, encoded, expected)
    circuit.simplify()
    verify_circuit(circuit, encoded, expected)
