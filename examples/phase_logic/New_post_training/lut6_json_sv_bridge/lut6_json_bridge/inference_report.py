"""Summarize clock-edge latency measured by the generated XSim testbench.

This file is copied into each generated graph project so run_sim.sh remains
self-contained and can be run without the bridge package on PYTHONPATH.
"""

from __future__ import annotations

import csv
from decimal import Decimal
import json
from pathlib import Path
import re


CASE = re.compile(
    r"^INFERENCE_CASE case=(\d+) input_to_pred_ns=([\d.]+) "
    r"start_to_pred_ns=([\d.]+)$", re.MULTILINE
)
PASS = re.compile(r"^ALL_TESTS_PASS cases=(\d+)$", re.MULTILINE)


def _cycles(duration_ns, clock_ns):
    measured = Decimal(duration_ns) / clock_ns
    cycles = int(measured.to_integral_value())
    if abs(measured - cycles) > Decimal("0.01"):
        raise ValueError(f"Latency is not an integer number of clocks: {duration_ns} ns")
    return cycles


def generate(project):
    project = Path(project)
    manifest = json.loads((project / "manifest.json").read_text())
    log = (project / "simulation.log").read_text()
    matches = list(CASE.finditer(log))
    passed = PASS.findall(log)
    count = len(manifest["reference_predictions"])
    if passed != [str(count)] or [int(m[1]) for m in matches] != list(range(count)):
        raise ValueError("XSim did not pass every expected inference case")

    clock_ns = Decimal(str(manifest["simulation_clock_ns"]))
    input_bits = manifest["input_bits"]
    rows = []
    for match in matches:
        case, input_ns, compute_ns = match.groups()
        input_cycles = _cycles(input_ns, clock_ns)
        compute_cycles = _cycles(compute_ns, clock_ns)
        if input_cycles - compute_cycles != input_bits:
            raise ValueError(f"Unexpected input-stream latency in case {case}")
        rows.append({
            "case": int(case),
            "predicted_class": manifest["reference_predictions"][int(case)],
            "start_to_prediction_cycles": compute_cycles,
            "start_to_prediction_ns": compute_ns,
            "input_to_prediction_cycles": input_cycles,
            "input_to_prediction_ns": input_ns,
        })

    report = project / "report_inference"
    report.mkdir(exist_ok=True)
    with (report / "latency.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    compute = [row["start_to_prediction_cycles"] for row in rows]
    total = [row["input_to_prediction_cycles"] for row in rows]
    def summary_line(label, values):
        mean_cycles = sum(values) / len(values)
        return (f"| {label} | {mean_cycles:,.0f} | "
                f"{mean_cycles * float(clock_ns) / 1e6:.6f} | "
                f"{min(values):,}–{max(values):,} |")

    checked = "class scores and prediction" if manifest["debug_scores_output"] else "prediction"
    lines = [
        "# XSim inference report", "",
        f"Functional checks: **PASS**, {count} cases ({checked}).", "",
        f"Testbench clock: **{clock_ns} ns**. Input: **{input_bits:,} pre-encoded bits**; "
        f"graph: **{manifest['graph_nodes']:,} nodes**, **{manifest['classes']} classes**.", "",
        "| Latency | Mean cycles | Mean simulated ms | Cycle range |",
        "| :-- | --: | --: | --: |",
        summary_line("Start accepted → prediction valid", compute),
        summary_line("First input bit accepted → prediction valid", total),
        "",
        f"Input streaming accounts for **{input_bits:,} cycles** "
        f"(**{input_bits * float(clock_ns) / 1e6:.6f} ms**).", "",
        "These are XSim clock times, not routed FPGA timing or host execution time. "
        "They exclude input encoding and external data transfer. The per-case values "
        "are in `latency.csv`.", "",
    ]
    (report / "summary.md").write_text("\n".join(lines))
    print(f"Inference report: {report / 'summary.md'}")


if __name__ == "__main__":
    generate(Path(__file__).resolve().parent.parent)
