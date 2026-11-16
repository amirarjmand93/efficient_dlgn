"""Summarize Vivado graph reports and compare runs on the same FPGA part."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUTS = ROOT.parent / "outputs"
RESOURCES = ("Slice LUTs", "Slice Registers", "Block RAM Tile", "DSPs", "Bonded IOB")
STAGES = ("synthesis", "placed", "routed")
MEMORY_NAMES = ("kind_mem", "rec1_mem", "rec2_mem", "rec3_mem", "rec4_mem",
                "rec5_mem", "rec6_mem", "truth_mem", "sum_addr_mem",
                "sum_inv_mem", "values_mem")


def read(path: Path) -> str:
    return path.read_text(errors="replace") if path.is_file() else ""


def header(report: str, name: str) -> str | None:
    match = re.search(rf"(?m)^\|\s*{re.escape(name)}\s*:\s*(.*?)\s*$", report)
    return match.group(1) if match else None


def table_value(report: str, name: str) -> str | None:
    for line in report.splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) >= 2 and cells[0].rstrip("*").strip() == name:
            return cells[1]
    return None


def number(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value.replace(",", ""))
    except ValueError:
        return None


def utilization(report: str) -> dict[str, float | None]:
    return {name: number(table_value(report, name)) for name in RESOURCES}


def timing(report: str) -> dict[str, float | None]:
    result = {"period_ns": None, "wns_ns": None, "critical_delay_ns": None,
              "logic_delay_ns": None, "routing_delay_ns": None}
    if not report:
        return result
    summary = report.split("| Design Timing Summary", 1)[-1]
    match = re.search(r"(?m)^\s*(-?\d+\.\d+)\s+-?\d+\.\d+\s+\d+\s+\d+\s+"
                      r"-?\d+\.\d+\s+-?\d+\.\d+\s+\d+\s+\d+", summary)
    if match:
        result["wns_ns"] = float(match.group(1))
    setup_paths = report.split("Max Delay Paths", 1)[-1]
    match = re.search(r"Data Path Delay:\s*(\d+(?:\.\d+)?)ns\s*"
                      r"\(logic\s*(\d+(?:\.\d+)?)ns.*?route\s*"
                      r"(\d+(?:\.\d+)?)ns", setup_paths)
    if match:
        for key, value in zip(("critical_delay_ns", "logic_delay_ns", "routing_delay_ns"),
                              match.groups()):
            result[key] = float(value)
    clock = report.split("| Clock Summary", 1)[-1]
    match = re.search(r"(?m)^\S+\s+\{[^}]+\}\s+(\d+(?:\.\d+)?)\s+\d", clock)
    if match:
        result["period_ns"] = float(match.group(1))
    return result


def route_errors(report: str) -> int | None:
    match = re.search(r"nets with routing errors\.*\s*:\s*(\d+)", report)
    return int(match.group(1)) if match else None


def bram_instances(path: Path) -> dict[str, float] | None:
    if not path.is_file():
        return None
    counts = defaultdict(float)
    for line in path.read_text(errors="replace").splitlines()[1:]:
        instance, _, primitive = line.rpartition(",")
        if not instance or not primitive.startswith(("RAMB36", "RAMB18")):
            continue
        name = next((memory for memory in MEMORY_NAMES if memory in instance), "other")
        counts[name] += 1.0 if primitive.startswith("RAMB36") else 0.5
    return dict(counts)


def inspect(folder: Path, outputs: Path) -> dict:
    reports = {stage: read(folder / f"{stage}_utilization.rpt") for stage in STAGES}
    source = next((text for text in reports.values() if text), "")
    manifest_path = folder.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    power = read(folder / "routed_power.rpt")
    bram_stage = ("synthesis" if (folder / "synthesis_bram_instances.csv").is_file()
                  else "routed" if (folder / "routed_bram_instances.csv").is_file()
                  else None)
    return {
        "folder": folder,
        "name": folder.relative_to(outputs).as_posix(),
        "device": header(source, "Device") or "unknown",
        "vivado": header(source, "Tool Version") or "unknown",
        "classes": manifest.get("classes"),
        "graph_nodes": manifest.get("graph_nodes"),
        "bram_instances": (bram_instances(folder / f"{bram_stage}_bram_instances.csv")
                           if bram_stage else None),
        "bram_stage": bram_stage,
        "resources": {stage: utilization(reports[stage]) if reports[stage] else None
                      for stage in STAGES},
        "timing": timing(read(folder / "routed_timing_summary.rpt")),
        "route_errors": route_errors(read(folder / "route_status.rpt")),
        "power_w": number(table_value(power, "Total On-Chip Power (W)")),
        "power_confidence": table_value(power, "Confidence Level"),
    }


def fmt(value: float | int | None, digits: int = 0) -> str:
    return "—" if value is None else f"{value:,.{digits}f}"


def bar(value: float | None, maximum: float, width: int = 20) -> str:
    if value is None or maximum <= 0:
        return "—"
    filled = max(1, round(width * value / maximum)) if value > 0 else 0
    return "█" * filled + "░" * (width - filled)


def link(target: Path, from_folder: Path, label: str) -> str:
    return f"[{label}](<{os.path.relpath(target, from_folder)}>)"


def stage_table(runs: list[dict], stage: str, from_folder: Path) -> list[str]:
    present = [run for run in runs if run["resources"][stage] is not None]
    if not present:
        return []
    lines = [f"### {stage.title()} utilization", "",
             "| Run | LUTs | Flip flops | BRAM tiles | DSPs | I/O pins |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for run in present:
        data = run["resources"][stage]
        values = [fmt(data[name], 1 if name == "Block RAM Tile" else 0)
                  for name in RESOURCES]
        lines.append(f"| {link(run['folder'] / 'analysis.md', from_folder, run['name'].split('/')[0])} "
                     f"| {' | '.join(values)} |")
    return lines + [""]


def individual(run: dict, outputs: Path) -> str:
    folder = run["folder"]
    lines = [f"# Vivado analysis: {run['name']}", "",
             f"- FPGA part: `{run['device']}`",
             f"- Tool: {run['vivado']}",
             f"- Model: {fmt(run['classes'])} classes; {fmt(run['graph_nodes'])} graph nodes", ""]
    stages = [stage for stage in STAGES if run["resources"][stage] is not None]
    if stages:
        lut_max = max(run["resources"][stage]["Slice LUTs"] or 0 for stage in stages)
        bram_max = max(run["resources"][stage]["Block RAM Tile"] or 0 for stage in stages)
        lines.extend(["## Utilization by stage", "",
                      "Bars scale to the largest stage value in this run.", "",
                      "| Stage | LUTs | Flip flops | BRAM tiles | DSPs | I/O pins |",
                      "| --- | --- | ---: | --- | ---: | ---: |"])
        for stage in stages:
            data = run["resources"][stage]
            luts, bram = data["Slice LUTs"], data["Block RAM Tile"]
            lines.append(f"| {stage.title()} | {bar(luts, lut_max)} {fmt(luts)} "
                         f"| {fmt(data['Slice Registers'])} "
                         f"| {bar(bram, bram_max)} {fmt(bram, 1)} "
                         f"| {fmt(data['DSPs'])} | {fmt(data['Bonded IOB'])} |")
        lines.append("")
    if run["bram_instances"] is not None:
        lines.extend([f"## {run['bram_stage'].title()} BRAM by inferred memory", "",
                      "Instance-name grouping; unmatched RAMB cells appear as `other`.", "",
                      "| Memory | BRAM tiles |", "| --- | ---: |"])
        for memory, count in sorted(run["bram_instances"].items(), key=lambda item: -item[1]):
            lines.append(f"| `{memory}` | {fmt(count, 1)} |")
        lines.extend([f"| **Total** | **{fmt(sum(run['bram_instances'].values()), 1)}** |", ""])
    t = run["timing"]
    lines.extend(["## Timing, routing, and power", "",
                  f"- Clock period: {fmt(t['period_ns'], 3)} ns",
                  f"- Critical setup data path delay: {fmt(t['critical_delay_ns'], 3)} ns "
                  f"(logic {fmt(t['logic_delay_ns'], 3)} ns; "
                  f"routing {fmt(t['routing_delay_ns'], 3)} ns)",
                  f"- Routing errors: {fmt(run['route_errors'])}",
                  f"- Estimated on-chip power: {fmt(run['power_w'], 3)} W "
                  f"(confidence: {run['power_confidence'] or 'unavailable'})", ""])
    if run["route_errors"] is None:
        lines.append("Routing is not verified by the available reports.")
    elif run["route_errors"] > 0:
        lines.append("Vivado reported routing errors; this implementation is incomplete.")
    elif t["wns_ns"] is not None and t["wns_ns"] < 0:
        lines.append("Routing completed, but the reported clock timing constraint was missed.")
    else:
        lines.append("Routing completed with zero reported errors.")
    lines.extend(["", "Power is a Vivado estimate. Board I/O timing is not constrained by these reports.", "",
                  "## Source reports", ""])
    for filename in ("synthesis_utilization.rpt", "synthesis_hierarchical_utilization.rpt",
                     "synthesis_bram_instances.csv", "routed_hierarchical_utilization.rpt",
                     "routed_bram_instances.csv", "placed_utilization.rpt",
                     "routed_utilization.rpt", "routed_timing_summary.rpt",
                     "route_status.rpt", "routed_power.rpt"):
        if (folder / filename).is_file():
            lines.append(f"- {link(folder / filename, folder, filename)}")
    return "\n".join(lines) + "\n"


def summary(runs: list[dict], outputs: Path, summary_path: Path) -> str:
    lines = ["# FPGA report comparison", "",
             f"Source: `{outputs}`. Runs are compared only within the same exact FPGA part.",
             "Missing reports are shown as —. Resource tables keep synthesis, placement, and routing separate.", ""]
    by_device = defaultdict(list)
    for run in runs:
        by_device[run["device"]].append(run)
    for device, group in sorted(by_device.items()):
        lines.extend([f"## {device}", ""])
        for stage in STAGES:
            lines.extend(stage_table(group, stage, summary_path.parent))
        memory_runs = [run for run in group if run["bram_instances"] is not None]
        if memory_runs:
            lines.extend(["### BRAM by inferred memory", "",
                          "Counts are BRAM tiles grouped by instance name; see each run for the synthesis or routed stage. `other` is unmatched.", "",
                          "| Run | " + " | ".join(MEMORY_NAMES) + " | Other |",
                          "| --- | " + " | ".join("---:" for _ in MEMORY_NAMES) + " | ---: |"])
            for run in memory_runs:
                counts = run["bram_instances"]
                values = [fmt(counts.get(memory, 0.0), 1) for memory in MEMORY_NAMES]
                lines.append(f"| {link(run['folder'] / 'analysis.md', summary_path.parent, run['name'].split('/')[0])} "
                             f"| {' | '.join(values)} | {fmt(counts.get('other', 0.0), 1)} |")
            lines.append("")
        present = [run for run in group if run["route_errors"] is not None
                   or run["timing"]["critical_delay_ns"] is not None or run["power_w"] is not None]
        if present:
            routed = [run for run in present if run["resources"]["routed"] is not None]
            if routed:
                lut_max = max(run["resources"]["routed"]["Slice LUTs"] or 0 for run in routed)
                bram_max = max(run["resources"]["routed"]["Block RAM Tile"] or 0 for run in routed)
                delay_max = max(run["timing"]["critical_delay_ns"] or 0 for run in routed)
                lines.extend(["### Routed visual comparison", "",
                              "Bars scale to the largest measured value for this FPGA part.", "",
                              "| Run | LUTs | BRAM tiles | Critical delay |",
                              "| --- | --- | --- | --- |"])
                for run in routed:
                    data = run["resources"]["routed"]
                    luts, bram = data["Slice LUTs"], data["Block RAM Tile"]
                    delay = run["timing"]["critical_delay_ns"]
                    lines.append(f"| {link(run['folder'] / 'analysis.md', summary_path.parent, run['name'].split('/')[0])} "
                                 f"| {bar(luts, lut_max)} {fmt(luts)} "
                                 f"| {bar(bram, bram_max)} {fmt(bram, 1)} "
                                 f"| {bar(delay, delay_max)} {fmt(delay, 3)} ns |")
                lines.append("")
            lines.extend(["### Routed timing, routing, and power", "",
                          "| Run | Clock ns | Critical delay ns | Logic ns | Routing ns | "
                          "Route errors | Power W | Power confidence |",
                          "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |"])
            for run in present:
                t = run["timing"]
                lines.append(f"| {link(run['folder'] / 'analysis.md', summary_path.parent, run['name'].split('/')[0])} "
                             f"| {fmt(t['period_ns'], 3)} | {fmt(t['critical_delay_ns'], 3)} "
                             f"| {fmt(t['logic_delay_ns'], 3)} | {fmt(t['routing_delay_ns'], 3)} "
                             f"| {fmt(run['route_errors'])} | {fmt(run['power_w'], 3)} "
                             f"| {run['power_confidence'] or '—'} |")
            lines.append("")
        incomplete = [run for run in group if run["route_errors"] is None]
        for run in incomplete:
            lines.append(f"- {link(run['folder'] / 'analysis.md', summary_path.parent, run['name'])}: "
                         "no routing report; implementation status is incomplete.")
        if incomplete:
            lines.append("")
    lines.extend(["Critical delay is the data path delay of Vivado's worst setup path, "
                  "not the complete clock period with skew and uncertainty.",
                  "Power estimates depend on activity assumptions; read each power report's confidence level.",
                  "I/O input and output delays were not constrained, so timing results cover the constrained paths only.", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs", type=Path, default=DEFAULT_OUTPUTS)
    parser.add_argument("--summary", type=Path, default=ROOT / "summary.md")
    args = parser.parse_args()
    outputs = args.outputs.resolve()
    if not outputs.is_dir():
        parser.error(f"Outputs directory does not exist: {outputs}")
    folders = sorted(folder for folder in outputs.rglob("reports_*")
                     if folder.is_dir() and any(folder.glob("*.rpt")))
    if not folders:
        parser.error(f"No Vivado report folders found under {outputs}")
    runs = [inspect(folder, outputs) for folder in folders]
    for run in runs:
        (run["folder"] / "analysis.md").write_text(individual(run, outputs))
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(summary(runs, outputs, args.summary.resolve()))
    print(f"Analyzed {len(runs)} report folders across {len(set(run['device'] for run in runs))} devices")
    print(f"Comparison: {args.summary.resolve()}")


if __name__ == "__main__":
    main()
