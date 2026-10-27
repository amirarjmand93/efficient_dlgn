"""Focused checkpoint selection and headless histogram checks."""

import csv

import pytest
import torch

from examples.phase_logic.plot_gate_histogram import load_gate_frequencies, save_histogram


@pytest.mark.parametrize("wrapped", [False, True])
def test_histogram_uses_actual_opcodes_and_pinned_rows(tmp_path, wrapped):
    state = {
        "1.impl.weights": torch.tensor([[0., 1., 3.], [5., 1., 0.]]),
        "1.impl.gate_sequence": torch.tensor([[3, 1, 9], [14, 5, 6]], dtype=torch.uint8),
        "1.impl.offsets_ch": torch.zeros(2, 2, 3, dtype=torch.int32),
        "6.impl.weights": torch.tensor([[0., 3.], [3., 0.], [0., 3.], [3., 0.]]),
        "6.impl.gate_sequence": torch.tensor([[3, 6], [9, 1], [5, 7], [14, 15]], dtype=torch.uint8),
        "6.impl.indices": torch.zeros(2, 4, 2, dtype=torch.long),
        "6.frozen_rows": torch.tensor([False, False, True, True]),
        "6.frozen_codes": torch.tensor([0, 0, 0, 15], dtype=torch.uint8),
    }
    path = tmp_path / "model.pt"
    torch.save({"model_state": state} if wrapped else state, path)
    layers = load_gate_frequencies(path)
    assert [layer["kind"] for layer in layers] == ["conv", "dense"]
    assert [layer["total"] for layer in layers] == [2, 4]
    assert layers[0]["counts"][9] == layers[0]["counts"][14] == 1
    assert all(layers[1]["counts"][i] == 1 for i in (0, 6, 9, 15))
    assert [layer["layer"] for layer in layers] == ["1", "6"]
    for layer in layers:
        assert sum(layer["counts"]) == layer["total"]


def test_headless_image_and_csv(tmp_path):
    pytest.importorskip("matplotlib")
    output = tmp_path / "plots" / "histogram.png"
    layers = [{"layer": "6", "kind": "dense", "label": "Dense 1 [6]",
               "counts": [1] * 16, "total": 16}]
    csv_path = save_histogram(layers, output, frequency="percent", dpi=50, checkpoint_name="test.pt")
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    with csv_path.open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 16
    assert sum(int(row["count"]) for row in rows) == 16
    assert sum(float(row["percent"]) for row in rows) == pytest.approx(100)


def test_unsupported_checkpoint_has_actionable_error(tmp_path):
    path = tmp_path / "other.pt"
    torch.save({"layer.weight": torch.ones(2, 2)}, path)
    with pytest.raises(ValueError, match="No CompactLogic"):
        load_gate_frequencies(path)
