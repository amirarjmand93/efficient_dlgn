#!/usr/bin/env python3
"""Replace Torchlogix GroupSum RTL with an exact, balanced 128-bit chunked reduction.

Designed for Torchlogix circuit.py at commit 82e5b48 and the provided
CompactLogic exporter. No torch/torchlogix installation needed.

The normal interface needs only the exported ``circuit.v``.  Its sibling
``circuit.json`` is discovered automatically and a new ``circuit.v`` is placed
under this script's ``generated/`` directory.  Explicit paths remain available.

This deliberately changes only the Verilog implementation of SumReduction;
the Boolean gates and all original raw[] connections remain untouched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re

from dense_locality import analyze_dense_connectivity, localize_dense_stages

CHUNK = 128
START_MARKER = "    // --- outputs (behavioral - synthesizer maps to carry chain) ---"
SCRIPT_DIR = Path(__file__).resolve().parent


def _sha256(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _default_output_dir(verilog: Path) -> Path:
    """Return a stable, separate destination inside this post-processing folder."""
    # Export paths normally end in <run-name>/circuit/circuit.v.  Include a
    # digest of the full source path so identically named exports cannot collide.
    run_name = verilog.parent.parent.name if verilog.parent.name == "circuit" else verilog.parent.name
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", run_name).strip("._") or "export"
    path_digest = hashlib.sha256(str(verilog.resolve()).encode("utf-8")).hexdigest()[:8]
    return SCRIPT_DIR / "generated" / f"{safe_name}_{path_digest}_hierarchical_groupsum"


def _popcount_module() -> str:
    """Exact 128-bit popcount, seven balanced pairwise-adder stages."""
    out = [
        "// 128-bit population-count reduction; 7 balanced levels, exact [0..128].",
        "module elar_popcount128 (",
        "    input  wire [127:0] bits,",
        "    output wire [7:0] count",
        ");",
        "    genvar i;",
    ]
    # Number of bits represented by each stage doubles each iteration.
    for level in range(1, 8):
        count = CHUNK // (2 ** level)
        width = level + 1
        prior = "bits" if level == 1 else f"s{level-1}"
        out.append(f"    wire [{width-1}:0] s{level} [0:{count-1}];")
        out.append("    generate")
        out.append(f"        for (i = 0; i < {count}; i = i + 1) begin : gen_l{level}")
        if level == 1:
            left = "bits[2*i]"
            right = "bits[2*i+1]"
        else:
            left = f"s{level-1}[2*i]"
            right = f"s{level-1}[2*i+1]"
        out.append(f"            assign s{level}[i] = {{1'b0, {left}}} + {{1'b0, {right}}};")
        out.append("        end")
        out.append("    endgenerate")
    out += ["    assign count = s7[0];", "endmodule", ""]
    return "\n".join(out)


def _validate_original_groupsum(old_tail: str, output_ids: list, sums: dict,
                                score_width: int) -> None:
    """Prove that the behavioral block and IR describe the same ordered sums."""
    statement = re.compile(
        r"s_(\d+)\s*=\s*(-?\d+)\s*;\s*"
        r"for\s*\(\s*i\s*=\s*(\d+)\s*;\s*i\s*<\s*(\d+)\s*;\s*"
        r"i\s*=\s*i\s*\+\s*1\s*\)\s*"
        r"s_\1\s*=\s*s_\1\s*\+\s*raw\[i\]\s*;\s*"
        r"scores_flat\[\s*(\d+)\s*\*\s*(\d+)\s*\s*\+:\s*(\d+)\s*\]\s*=\s*s_\1\s*;"
    )
    blocks = statement.findall(old_tail)
    if len(blocks) != len(output_ids):
        raise ValueError("Could not identify one original GroupSum loop per output class.")
    offset = 0
    for class_index, (sum_index, beta, lo, hi, bus_index, stride, width) in enumerate(blocks):
        sr = sums[int(output_ids[class_index])]
        expected_beta = float(sr.get("beta", 0.0))
        if not expected_beta.is_integer():
            raise ValueError(f"Output {class_index}: original GroupSum beta is not integral.")
        expected_hi = offset + len(sr["input_ids"])
        actual = tuple(map(int, (sum_index, beta, lo, hi, bus_index, stride, width)))
        expected = (class_index, int(expected_beta), offset, expected_hi,
                    class_index, score_width, score_width)
        if actual != expected:
            raise ValueError(
                f"Original GroupSum output {class_index} differs from circuit.json: "
                f"RTL {actual}, expected {expected}."
            )
        offset = expected_hi


def _assemble(ir: dict, original: str, keep_hierarchy: bool) -> tuple[str, dict]:
    n_inputs = int(ir["n_inputs"])
    output_ids = ir["outputs"]
    sums = {int(sr["node_id"]): sr for sr in ir.get("sum_nodes", [])}
    if not output_ids or any(int(x) not in sums for x in output_ids):
        raise ValueError("Expected all output IDs to be SumReduction nodes (10 class counts).")

    if original.count(START_MARKER) != 1:
        raise ValueError("Original Verilog does not match the pinned Torchlogix GroupSum emitter.")
    prefix, _, old_tail = original.partition(START_MARKER)
    if not re.fullmatch(r"\s*integer\s+[^;]+;\s*always\s*@\(\*\)\s*begin\s*.*?\s*end\s+endmodule\s*", old_tail, re.S):
        raise ValueError("Refusing to replace an unfamiliar Verilog output block.")
    if "endmodule" in prefix or original.count("endmodule") != 1:
        raise ValueError("Expected a single Verilog module with no additional modules.")
    port = re.search(r"\boutput\s+reg\s*\[\s*(\d+)\s*:\s*0\s*\]\s*scores_flat\b", prefix)
    if port is None:
        raise ValueError("Cannot find output reg [TOTAL-1:0] scores_flat in original RTL.")
    total_width = int(port.group(1)) + 1
    if total_width % len(output_ids):
        raise ValueError("Score bus width is not divisible by the class count.")
    score_width = total_width // len(output_ids)
    if score_width not in (8, 16, 32, 64):
        raise ValueError(f"Unsupported score width {score_width}.")
    _validate_original_groupsum(old_tail, output_ids, sums, score_width)
    prefix = prefix[:port.start()] + port.group(0).replace("output reg", "output wire", 1) + prefix[port.end():]

    expected_raw = []
    for oid in output_ids:
        expected_raw.extend(int(g) for g in sums[int(oid)]["input_ids"])
    declared = re.search(r"\bwire\s*\[\s*(\d+)\s*:\s*0\s*\]\s*raw\s*;", prefix)
    if expected_raw:
        if declared is None or int(declared.group(1)) + 1 != len(expected_raw):
            raise ValueError("raw[] width differs from circuit.json; perhaps files are from different exports.")
    raw_assigns = re.findall(r"\bassign\s+raw\[(\d+)\]\s*=\s*(inp\[\d+\]|g\d+)\s*;", prefix)
    if len(raw_assigns) != len(expected_raw):
        raise ValueError("Raw-input assignments differ from expected non-inlined exporter format.")
    for idx, ((number, name), gid) in enumerate(zip(raw_assigns, expected_raw)):
        expected_name = f"inp[{gid}]" if gid < n_inputs else f"g{gid}"
        if int(number) != idx or name != expected_name:
            raise ValueError(f"RTL and JSON differ at raw[{idx}]: {name} vs {expected_name}.")

    parts = [
        "    // --- hierarchical GroupSum, exact integer class counts ---",
        "    // 128-bit chunks; each local popcount uses a balanced 7-level tree.",
        "    // No new registers, no clocks, and no changes to logic gate connections.",
    ]
    stats = {"class_count": len(output_ids), "score_width": score_width,
             "chunk_size": CHUNK, "total_chunk_instances": 0, "per_class": [],
             "source_prefix_sha256": _sha256(prefix)}
    offset = 0
    for class_index, oid in enumerate(output_ids):
        sr = sums[int(oid)]
        if float(sr.get("tau", 1.0)) != 1.0:
            raise ValueError(f"Output {class_index}: requires tau=1 raw count.")
        beta = float(sr.get("beta", 0.0))
        if not math.isfinite(beta) or not beta.is_integer() or beta < 0:
            raise ValueError(f"Output {class_index}: beta must be a nonnegative integer.")
        beta = int(beta)
        nbits = len(sr["input_ids"])
        if nbits + beta >= (1 << score_width):
            raise ValueError(f"Output {class_index}: exact count may overflow the {score_width}-bit port.")
        parts.append(f"    // Class {class_index}: {nbits} raw bits plus folded constant {beta}.")
        local_nodes = []
        for chunk_index, chunk_start in enumerate(range(0, nbits, CHUNK)):
            take = min(CHUNK, nbits - chunk_start)
            lo = offset + chunk_start
            hi = lo + take - 1
            source = f"raw[{lo}]" if take == 1 else f"raw[{hi}:{lo}]"
            if take < CHUNK:
                source = "{" + f"{CHUNK-take}'b0, {source}" + "}"
            name = f"gs_c{class_index}_chunk{chunk_index}"
            parts.append(f"    wire [7:0] {name};")
            attr = '(* keep_hierarchy = "yes" *) ' if keep_hierarchy else ""
            parts.append(f"    {attr}elar_popcount128 pc_c{class_index}_{chunk_index} ("
                         f".bits({source}), .count({name}));")
            local_nodes.append((name, 8))
        stats["total_chunk_instances"] += len(local_nodes)
        stats["per_class"].append({"class_index": class_index,
                                  "boolean_inputs": nbits, "folded_ones": beta,
                                  "chunks": len(local_nodes)})
        level = 1
        while len(local_nodes) > 1:
            nxt = []
            for i in range(0, len(local_nodes), 2):
                if i + 1 == len(local_nodes):
                    nxt.append(local_nodes[i])
                    continue
                (a, aw), (b, bw) = local_nodes[i:i+2]
                width = max(aw, bw) + 1
                node = f"gs_c{class_index}_l{level}_{i//2}"
                parts.append(f"    wire [{width-1}:0] {node} = "
                             f"{{1'b0, {a}}} + {{1'b0, {b}}};")
                nxt.append((node, width))
            local_nodes = nxt
            level += 1
        count_expr = local_nodes[0][0] if local_nodes else "1'b0"
        if beta:
            parts.append(f"    assign scores_flat[{class_index}*{score_width} +: {score_width}] = "
                         f"{count_expr} + {score_width}'d{beta};")
        else:
            parts.append(f"    assign scores_flat[{class_index}*{score_width} +: {score_width}] = "
                         f"{count_expr};")
        offset += nbits
    parts += ["", "endmodule", "", _popcount_module()]
    return prefix + "\n".join(parts), stats


def _resolve_paths(args: argparse.Namespace, parser: argparse.ArgumentParser) -> tuple[Path, Path, Path]:
    if args.circuit_v is not None and args.legacy_verilog is not None:
        parser.error("Pass circuit.v positionally or with --verilog, not both.")
    verilog = args.circuit_v or args.legacy_verilog
    if verilog is None:
        parser.error("A circuit.v input is required.")
    verilog = verilog.expanduser().resolve()
    ir = (args.ir.expanduser().resolve() if args.ir is not None
          else verilog.with_name("circuit.json"))
    if args.output is not None and args.output_dir is not None:
        parser.error("Choose --output or --output-dir, not both.")
    if args.output is not None:
        output = args.output.expanduser().resolve()
    else:
        output_dir = (args.output_dir.expanduser().resolve() if args.output_dir is not None
                      else _default_output_dir(verilog))
        output = output_dir / "circuit.v"
    if verilog == output:
        parser.error("Output must differ from the original Verilog file; retain a baseline.")
    if not verilog.is_file():
        parser.error(f"Verilog input does not exist: {verilog}")
    if not ir.is_file():
        parser.error(f"Circuit IR does not exist: {ir} (use --ir to select it explicitly)")
    if output.exists() and not args.force:
        parser.error(f"Output already exists: {output} (use --force to replace the generated copy)")
    return verilog, ir, output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("circuit_v", nargs="?", type=Path,
                        help="Original circuit.v; sibling circuit.json is used by default")
    parser.add_argument("--ir", type=Path, help="Original circuit.json (default: beside circuit.v)")
    parser.add_argument("--verilog", dest="legacy_verilog", type=Path,
                        help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path, help="Exact new circuit.v destination")
    parser.add_argument("--output-dir", type=Path,
                        help="Separate destination directory (file name remains circuit.v)")
    parser.add_argument("--keep-hierarchy", action="store_true",
                        help="Attempt to preserve individual 128-bit popcount instances in Vivado; test separately")
    parser.add_argument("--localize-dense", action="store_true",
                        help="Co-locate connected gates in the final two Boolean stages")
    parser.add_argument("--dense-tile-size", type=int, default=512,
                        help="Approximate gates per connectivity tile (default: 512)")
    parser.add_argument("--force", action="store_true",
                        help="Replace an existing generated output; the source is still protected")
    args = parser.parse_args()
    verilog, ir_path, output = _resolve_paths(args, parser)
    try:
        ir = json.loads(ir_path.read_text(encoding="utf-8"))
        original = verilog.read_text(encoding="utf-8")
        modified, info = _assemble(ir, original, args.keep_hierarchy)
        locality_info = None
        connectivity_info = None
        if args.localize_dense:
            connectivity_info = analyze_dense_connectivity(ir)
            modified, locality_info = localize_dense_stages(
                ir, modified, target_size=args.dense_tile_size
            )
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        parser.error(str(error))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(modified, encoding="utf-8")
    report = {
        "source_verilog": str(verilog),
        "source_ir": str(ir_path),
        "output_verilog": str(output),
        "source_verilog_sha256": _sha256(original),
        "output_verilog_sha256": _sha256(modified),
        "top_module": "circuit",
        "combinational_only": True,
        "keep_hierarchy": args.keep_hierarchy,
        "groupsum": info,
        "connectivity_analysis": connectivity_info,
        "dense_locality": locality_info,
    }
    report_path = output.with_name("transformation_report.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Created {output} (top module: circuit)")
    print(f"GroupSum: {info['class_count']} classes, {info['total_chunk_instances']} 128-bit blocks,"
          f" {info['score_width']}-bit outputs")
    if locality_info is not None:
        print(f"Dense locality: {locality_info['tile_count']} tiles, "
              f"{locality_info['dense1_to_dense2_edges']} internal Dense-1 -> Dense-2 edges, "
              "0 crossing tile boundaries")
    print(f"Audit report: {report_path}")
    print("Use verification_vectors.npz and RTL simulation to verify exact class counts before synthesis.")


if __name__ == "__main__":
    main()
