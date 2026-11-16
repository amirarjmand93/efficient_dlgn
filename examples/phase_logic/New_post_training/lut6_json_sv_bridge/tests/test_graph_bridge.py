"""Exact teacher graph compilation and compressed ROM regression checks."""

import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lut6_json_bridge.graph_compiler import (
    build, compile_graph, evaluate_native, evaluate_program, predicted_class,
)


def _signal(source, inverted=False):
    return {"source": source, "inverted": inverted}


def _gate(gate_id, pins, init):
    return {"gate_id": gate_id, "inputs": [_signal(x) for x in pins]
            + [_signal(-1)] * (6 - len(pins)), "init_hex": f"{init:016X}"}


def _graph():
    # AND -> XOR uses a previous gate, plus duplicate/inverted GroupSum inputs.
    return {
        "format": "torchlogix_native_lut6_circuit_v1",
        "input_shape": [3], "score_bits": 5,
        "metadata": {"pin0_is_address_lsb": True},
        "gates": [
            _gate(7, [0, 1], 0x8888888888888888),
            _gate(9, [7, 2], 0x6666666666666666),
            _gate(10, [0], 0x5555555555555555),
        ],
        "sums": [
            {"signals": [_signal(9), _signal(7), _signal(9, True)], "offset": 2},
            {"signals": [_signal(10), _signal(1), _signal(1)], "offset": 1},
        ],
    }


def test_graph_compression_and_exact_scores(tmp_path):
    graph = _graph()
    spec = compile_graph(graph)
    assert spec["n_program"] == 3
    assert spec["n_values"] == 6
    assert spec["instructions"][1][0] == (3, 2)
    for pattern in range(8):
        bits = [(pattern >> pin) & 1 for pin in range(3)]
        assert evaluate_program(spec, bits) == evaluate_native(graph, bits)
    path = tmp_path / "teacher.json"
    path.write_text(json.dumps(graph))
    manifest = build(path, tmp_path / "project", random_tests=2)
    assert manifest["distinct_truth_tables"] == 3
    assert manifest["reference_scores"][0] == [3, 2]
    rtl = (tmp_path / "project/rtl/lut6_sequential.sv").read_text()
    assert "__" not in rtl
    assert "sum_addr_mem" in rtl and "values_mem" in rtl


def test_rejects_forward_reference():
    graph = _graph()
    graph["gates"][0]["inputs"][0]["source"] = 9
    with pytest.raises(ValueError, match="future or missing"):
        compile_graph(graph)


@pytest.mark.parametrize("classes,width", [(10, 4), (100, 7), (13, 4), (16, 4)])
def test_graph_argmax_output(tmp_path, classes, width):
    graph = _graph()
    # Zero input ties all classes at zero. One input ties classes 2 and last
    # at the largest nonzero score, so the lower index must win.
    graph["sums"] = [
        {"signals": [_signal(0)] * (2 if cls in (2, classes - 1) else 1),
         "offset": 0}
        for cls in range(classes)
    ]
    ir = tmp_path / "teacher.json"
    ir.write_text(json.dumps(graph))
    out = tmp_path / "project"
    manifest = build(ir, out, random_tests=0)
    assert manifest["class_width_bits"] == width
    assert manifest["reference_predictions"][:2] == [0, 2]
    assert predicted_class([3, 5, 5]) == 1
    rtl = (out / "rtl/lut6_sequential.sv").read_text()
    assert "output reg pred_valid" in rtl
    assert f"output reg [{width}-1:0] predicted_class" in rtl
    assert "scores_flat" not in rtl
    tb = (out / "sim/tb_lut6_sequential.sv").read_text()
    assert "pred_valid" in tb and "golden_class" in tb
    if shutil.which("xvlog") and classes != 16:
        result = subprocess.run(["bash", "run_sim.sh"], cwd=out,
                                capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, result.stdout + result.stderr
        assert "ALL_TESTS_PASS" in (out / "simulation.log").read_text()


def test_graph_debug_scores_port(tmp_path):
    ir = tmp_path / "teacher.json"
    ir.write_text(json.dumps(_graph()))
    out = tmp_path / "project"
    manifest = build(ir, out, random_tests=0, debug_scores=True)
    assert manifest["debug_scores_output"] is True
    rtl = (out / "rtl/lut6_sequential.sv").read_text()
    tb = (out / "sim/tb_lut6_sequential.sv").read_text()
    assert "scores_flat" in rtl and "Score mismatch" in tb
    if shutil.which("xvlog"):
        result = subprocess.run(["bash", "run_sim.sh"], cwd=out,
                                capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, result.stdout + result.stderr
