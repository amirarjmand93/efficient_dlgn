"""Compile a native teacher LUT6 DAG into compressed sequential SystemVerilog.

The existing layer compiler remains unchanged. This path accepts
torchlogix_native_lut6_circuit_v1 and preserves its complete graph and sums.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random

from .compiler import IRValidationError, require


BASE = Path(__file__).resolve().parents[1]
TEMPLATE = BASE / "rtl" / "lut6_graph_engine.sv.in"


def _bits(count):
    return max(1, (count - 1).bit_length())


def _hex_lines(path, values, width):
    path.write_text("".join(f"{value:0{(width + 3) // 4}X}\n" for value in values))


def _normalize_init(init, signals):
    """Move active pins to low addresses, absorbing pin inversions exactly."""
    active = [index for index, signal in enumerate(signals) if signal["source"] >= 0]
    if active == list(range(len(active))) and not any(s["inverted"] for s in signals):
        return init, active
    result = 0
    for address in range(64):
        original = 0
        for bit, signal in enumerate(signals):
            value = bool((address >> active.index(bit)) & 1) if bit in active else False
            original |= int(value ^ signal["inverted"]) << bit
        result |= ((init >> original) & 1) << address
    return result, active


def compile_graph(ir):
    require(ir.get("format") == "torchlogix_native_lut6_circuit_v1",
            "Expected teacher native LUT6 graph JSON")
    shape = ir.get("input_shape")
    require(isinstance(shape, list) and len(shape) in (1, 3)
            and all(type(n) is int and n > 0 for n in shape), "Invalid Boolean input shape")
    n_input = math.prod(shape)
    gates, sums = ir.get("gates"), ir.get("sums")
    require(isinstance(gates, list) and gates and isinstance(sums, list) and sums,
            "Graph needs LUT6 gates and GroupSum reductions")
    require(ir.get("metadata", {}).get("pin0_is_address_lsb") is True,
            "Graph must declare pin0 as LUT address LSB")

    address_map = {}
    instructions = []
    truth_ids = {}
    truth_tables = []
    for index, gate in enumerate(gates):
        original_id = gate.get("gate_id")
        require(type(original_id) is int and original_id >= n_input
                and original_id not in address_map, f"Invalid gate ID at {index}")
        signals = gate.get("inputs")
        require(isinstance(signals, list) and len(signals) == 6,
                f"Gate {original_id} needs six pin descriptors")
        for signal in signals:
            require(type(signal.get("source")) is int and type(signal.get("inverted")) is bool,
                    f"Gate {original_id} has an invalid pin")
            source = signal["source"]
            require(source == -1 or 0 <= source < n_input or source in address_map,
                    f"Gate {original_id} reads a future or missing gate {source}")
        try:
            hex_init = gate["init_hex"]
            require(isinstance(hex_init, str) and len(hex_init) == 16,
                    f"Gate {original_id} INIT must have 16 hex digits")
            init = int(hex_init, 16)
        except (KeyError, ValueError) as error:
            raise IRValidationError(f"Gate {original_id} has invalid INIT") from error
        init, active = _normalize_init(init, signals)
        require(1 <= len(active) <= 6,
                f"Gate {original_id} must have one to six active pins")
        pins = tuple(source if source < n_input else address_map[source]
                     for source in (signals[pos]["source"] for pos in active))
        if init not in truth_ids:
            truth_ids[init] = len(truth_tables)
            truth_tables.append(init)
        instructions.append((pins, truth_ids[init]))
        address_map[original_id] = n_input + index

    sum_entries, lengths, offsets = [], [], []
    for cls, reduction in enumerate(sums):
        signals = reduction.get("signals")
        offset = reduction.get("offset")
        require(isinstance(signals, list) and signals and type(offset) is int and offset >= 0,
                f"Class {cls} has an invalid GroupSum")
        lengths.append(len(signals))
        offsets.append(offset)
        for signal in signals:
            source, inverted = signal.get("source"), signal.get("inverted")
            require(type(source) is int and type(inverted) is bool
                    and (0 <= source < n_input or source in address_map),
                    f"Class {cls} has a missing GroupSum input")
            sum_entries.append((source if source < n_input else address_map[source], inverted))

    n_values = n_input + len(instructions)
    addr_w = _bits(n_values)
    id_w = _bits(len(truth_tables))
    score_w = max(int(ir.get("score_bits", 1)),
                  max(length + offset for length, offset in zip(lengths, offsets)).bit_length())
    require(addr_w <= 30 and len(sums) <= 256, "Graph exceeds bridge address/class widths")
    return dict(shape=shape, n_input=n_input, n_values=n_values,
                n_program=len(instructions), instructions=instructions,
                truth_tables=truth_tables, sums=sum_entries,
                lengths=lengths, offsets=offsets, n_classes=len(sums),
                addr_w=addr_w, id_w=id_w, score_w=score_w,
                class_w=_bits(len(sums)), sum_w=_bits(len(sum_entries)),
                pc_w=_bits(len(instructions)))


def evaluate_program(spec, bits):
    require(len(bits) == spec["n_input"] and all(bit in (0, 1) for bit in bits),
            "Wrong Boolean input vector")
    values = list(bits)
    for pins, truth_id in spec["instructions"]:
        address = sum(values[source] << pin for pin, source in enumerate(pins))
        values.append((spec["truth_tables"][truth_id] >> address) & 1)
    scores = list(spec["offsets"])
    index = 0
    for cls, length in enumerate(spec["lengths"]):
        for source, inverted in spec["sums"][index:index + length]:
            scores[cls] += values[source] ^ inverted
        index += length
    return scores


def evaluate_native(ir, bits):
    """Independent evaluation using original graph IDs and six-pin INITs."""
    values = {index: bit for index, bit in enumerate(bits)}
    for gate in ir["gates"]:
        address = 0
        for pin, signal in enumerate(gate["inputs"]):
            raw = values[signal["source"]] if signal["source"] >= 0 else 0
            address |= (raw ^ signal["inverted"]) << pin
        values[gate["gate_id"]] = (int(gate["init_hex"], 16) >> address) & 1
    return [sum(values[s["source"]] ^ s["inverted"] for s in reduction["signals"])
            + reduction["offset"] for reduction in ir["sums"]]


def predicted_class(scores):
    """Return the first class with the largest exact integer score."""
    return max(range(len(scores)), key=scores.__getitem__)


def _bram_estimate(width, depth):
    # RAMB36 configurations; estimate only, actual utilization needs synthesis.
    return min(math.ceil(width / w) * math.ceil(depth / d)
               for w, d in ((1,32768),(2,16384),(4,8192),(9,4096),
                            (18,2048),(36,1024),(72,512)))


def write_roms(spec, out):
    rtl = out / "rtl"
    rtl.mkdir(parents=True, exist_ok=True)
    counts = Counter(len(pins) for pins, _ in spec["instructions"])
    kind = [len(pins) - 1 for pins, _ in spec["instructions"]]
    _hex_lines(rtl / "lut6_kind.mem", kind, 3)
    memories = {"kind": (3, len(kind))}
    for arity in range(1, 7):
        width = arity * spec["addr_w"] + spec["id_w"]
        words = []
        for pins, truth_id in spec["instructions"]:
            if len(pins) != arity:
                continue
            word = truth_id << (arity * spec["addr_w"])
            for index, source in enumerate(pins):
                word |= source << (index * spec["addr_w"])
            words.append(word)
        _hex_lines(rtl / f"lut6_rec{arity}.mem", words or [0], width)
        memories[f"rec{arity}"] = (width, max(1, len(words)))
    _hex_lines(rtl / "lut6_truth.mem", spec["truth_tables"], 64)
    memories["truth"] = (64, len(spec["truth_tables"]))
    addresses = [source for source, _ in spec["sums"]]
    packed_addr = [addresses[i] | ((addresses[i + 1] if i + 1 < len(addresses) else 0)
                                    << spec["addr_w"])
                   for i in range(0, len(addresses), 2)]
    _hex_lines(rtl / "lut6_sum_addr.mem", packed_addr, 2 * spec["addr_w"])
    memories["sum_addr"] = (2 * spec["addr_w"], len(packed_addr))
    inverted = [int(value) for _, value in spec["sums"]]
    packed_inv = [sum(inverted[i + j] << j for j in range(min(16, len(inverted) - i)))
                  for i in range(0, len(inverted), 16)]
    _hex_lines(rtl / "lut6_sum_inv.mem", packed_inv, 16)
    memories["sum_inv"] = (16, len(packed_inv))
    memories["values"] = (1, spec["n_values"])
    return memories, counts


def verify_rom_roundtrip(spec, out):
    """Decode the emitted memory files, including GroupSum order and offsets."""
    rtl = out / "rtl"
    kind = [int(line, 16) for line in (rtl / "lut6_kind.mem").read_text().splitlines()]
    records = {arity: [int(line, 16) for line in
                       (rtl / f"lut6_rec{arity}.mem").read_text().splitlines()]
               for arity in range(1, 7)}
    truth = [int(line, 16) for line in (rtl / "lut6_truth.mem").read_text().splitlines()]
    require(truth == spec["truth_tables"] and len(kind) == spec["n_program"],
            "Truth or kind ROM differs after serialization")
    pointers = {arity: 0 for arity in range(1, 7)}
    mask = (1 << spec["addr_w"]) - 1
    for index, (pins, truth_id) in enumerate(spec["instructions"]):
        arity = kind[index] + 1
        require(arity == len(pins), f"Gate {index} arity differs in ROM")
        word = records[arity][pointers[arity]]
        pointers[arity] += 1
        decoded = tuple((word >> (slot * spec["addr_w"])) & mask
                        for slot in range(arity))
        decoded_id = word >> (arity * spec["addr_w"])
        require((decoded, decoded_id) == (pins, truth_id),
                f"Gate {index} connection or truth ID differs in ROM")
    addr_words = [int(line, 16) for line in
                  (rtl / "lut6_sum_addr.mem").read_text().splitlines()]
    inv_words = [int(line, 16) for line in
                 (rtl / "lut6_sum_inv.mem").read_text().splitlines()]
    for index, expected in enumerate(spec["sums"]):
        source = (addr_words[index // 2] >> ((index % 2) * spec["addr_w"])) & mask
        inverted = bool((inv_words[index // 16] >> (index % 16)) & 1)
        require((source, inverted) == expected,
                f"GroupSum entry {index} differs in ROM")


def render_rtl(spec, out, *, debug_scores=False):
    replacements = {
        "__N_INPUT__": spec["n_input"], "__N_PROGRAM__": spec["n_program"],
        "__N_VALUES__": spec["n_values"], "__N_TRUTH__": len(spec["truth_tables"]),
        "__N_SUM__": len(spec["sums"]), "__N_CLASSES__": spec["n_classes"],
        "__ADDR_W__": spec["addr_w"], "__ID_W__": spec["id_w"],
        "__SCORE_W__": spec["score_w"], "__CLASS_W__": spec["class_w"],
        "__SUM_W__": spec["sum_w"], "__PC_W__": spec["pc_w"],
        "__LOAD_W__": _bits(spec["n_input"]),
    }
    text = TEMPLATE.read_text()
    if debug_scores:
        text = text.replace("__SCORES_PORT__",
                            ",\n    output wire [__N_CLASSES__*__SCORE_W__-1:0] scores_flat")
        text = text.replace("__SCORES_ASSIGN__", '''genvar g;
    generate for(g=0;g<N_CLASSES;g=g+1) begin: output_scores
        assign scores_flat[g*SCORE_W +: SCORE_W]=class_counts[g];
    end endgenerate''')
    else:
        text = text.replace("__SCORES_PORT__", "")
        text = text.replace("__SCORES_ASSIGN__", "")
    for token, value in replacements.items():
        text = text.replace(token, str(value))
    record_code = []
    select_code = []
    reset_code = []
    advance_code = []
    for arity in range(1, 7):
        index = arity - 1
        width = arity * spec["addr_w"] + spec["id_w"]
        length = max(1, sum(len(pins) == arity for pins, _ in spec["instructions"]))
        record_code.append(f'''    (* rom_style="block" *) reg [{width-1}:0] rec{arity}_mem[0:{length-1}];
    reg [{width-1}:0] rec{arity}_q;
    reg [PC_W-1:0] rec{arity}_index;
    initial $readmemh("lut6_rec{arity}.mem", rec{arity}_mem);
    always @(posedge clk) if (state==S_REC && kind_q==3'd{index})
        rec{arity}_q <= rec{arity}_mem[rec{arity}_index];''')
        select_code.append(f"            3'd{index}: rec_word = {{{{(REC_MAX_W-{width}){{1'b0}}}},rec{arity}_q}};")
        reset_code.append(f"                        rec{arity}_index <= 0;")
        advance_code.append(f"                        3'd{index}: rec{arity}_index <= rec{arity}_index + 1'b1;")
    text = text.replace("__RECORD_MEMORIES__", "\n".join(record_code))
    text = text.replace("__RECORD_SELECT__", "\n".join(select_code))
    text = text.replace("__RECORD_RESET__", "\n".join(reset_code))
    text = text.replace("__RECORD_ADVANCE__", "\n".join(advance_code))
    text = text.replace("__CLASS_LENGTH__", "\n".join(
        f"            {spec['class_w']}'d{i}: class_length = {spec['sum_w']}'d{length};"
        for i, length in enumerate(spec["lengths"])))
    text = text.replace("__CLASS_OFFSET__", "\n".join(
        f"            {spec['class_w']}'d{i}: class_offset = {spec['score_w']}'d{offset};"
        for i, offset in enumerate(spec["offsets"])))
    require("__" not in text, "Unreplaced graph RTL template placeholder")
    (out / "rtl" / "lut6_sequential.sv").write_text(text)


def render_tb(spec, patterns, scores, out, *, debug_scores=False):
    vectors = out / "vectors"
    vectors.mkdir(exist_ok=True)
    for index, pattern in enumerate(patterns):
        (vectors / f"case_{index:03d}.mem").write_text("".join(f"{bit}\n" for bit in pattern))
    flat = [sum(value << (cls * spec["score_w"]) for cls, value in enumerate(score))
            for score in scores]
    _hex_lines(vectors / "golden.mem", flat, spec["n_classes"] * spec["score_w"])
    (vectors / "golden.json").write_text(json.dumps(scores, indent=2) + "\n")
    classes = [predicted_class(score) for score in scores]
    _hex_lines(vectors / "golden_class.mem", classes, spec["class_w"])
    (vectors / "golden_classes.json").write_text(json.dumps(classes, indent=2) + "\n")
    cases = "\n".join(f'            {index}: $readmemb("vectors/case_{index:03d}.mem", bits);'
                      for index in range(len(patterns)))
    tb = f'''`timescale 1ns/1ps
module tb_lut6_sequential;
  localparam N_INPUT={spec["n_input"]}, N_CASES={len(patterns)};
  localparam SCORE_W={spec["score_w"]}, N_CLASSES={spec["n_classes"]};
  localparam CLASS_W={spec["class_w"]};
  reg clk=0, rst=1, input_valid=0, input_bit=0, start=0;
  wire input_ready, frame_ready, busy, done, pred_valid;
  wire [CLASS_W-1:0] predicted_class;
{f'  wire [SCORE_W*N_CLASSES-1:0] scores_flat;' if debug_scores else ''}
  reg bits[0:N_INPUT-1];
  reg [CLASS_W-1:0] golden_class[0:N_CASES-1];
{f'  reg [SCORE_W*N_CLASSES-1:0] golden[0:N_CASES-1];' if debug_scores else ''}
  integer test_id, i;
  always #5 clk=~clk;
  lut6_sequential dut(.*);
  initial begin
    $readmemh("vectors/golden_class.mem",golden_class);
{f'    $readmemh("vectors/golden.mem",golden);' if debug_scores else ''}
    repeat(3) @(negedge clk);
    rst=0;
    for (test_id=0; test_id<N_CASES; test_id=test_id+1) begin
      case(test_id)
{cases}
      endcase
      for (i=0;i<N_INPUT;i=i+1) begin
        @(negedge clk);
        if (!input_ready) $fatal(1,"Input not ready");
        input_valid=1; input_bit=bits[i];
      end
      @(negedge clk); input_valid=0;
      if (!frame_ready) $fatal(1,"Frame not ready");
      start=1;
      @(negedge clk); start=0;
      wait(done===1'b1);
      if (pred_valid !== 1'b1)
        $fatal(1,"Prediction invalid at done, case=%0d",test_id);
      if (predicted_class !== golden_class[test_id])
        $fatal(1,"Class mismatch case=%0d got=%0d want=%0d",test_id,predicted_class,golden_class[test_id]);
{'''      if (scores_flat !== golden[test_id])
        $fatal(1,"Score mismatch case=%0d got=%h want=%h",test_id,scores_flat,golden[test_id]);''' if debug_scores else ''}
      $display("PASS case=%0d",test_id);
      @(negedge clk);
      @(negedge clk);
      if (pred_valid !== 1'b0)
        $fatal(1,"Prediction valid did not clear after done, case=%0d",test_id);
    end
    $display("PREDICTIONS_PASS cases=%0d",N_CASES);
    $display("ALL_TESTS_PASS cases=%0d",N_CASES);
    $finish;
  end
  initial begin #1000000000; $fatal(1,"Graph simulation watchdog"); end
endmodule
'''
    (out / "sim" / "tb_lut6_sequential.sv").write_text(tb)


def write_support(out, clock_ns):
    (out / "timing.xdc").write_text(f"create_clock -name core_clk -period {clock_ns:g} [get_ports clk]\n")
    vivado = out / "vivado"
    vivado.mkdir(exist_ok=True)
    (vivado / "implement.tcl").write_text('''# vivado -mode batch -source vivado/implement.tcl -tclargs <part> <tag>
set root [file normalize [file join [file dirname [info script]] ..]]
set part [lindex $argv 0]
set tag [lindex $argv 1]
if {$part eq ""} { error "Supply a target FPGA part" }
if {$tag eq ""} { set tag default }
set outdir [file join $root reports_$tag]
file mkdir $outdir
create_project -force "teacher_lut6_$tag" [file join $root vivado_build_$tag] -part $part
add_files -norecurse [file join $root rtl lut6_sequential.sv]
set mem_files [glob [file join $root rtl *.mem]]
if {[llength $mem_files] != 10} {
    error "Expected all ten LUT6 ROM initialization files"
}
foreach mem $mem_files {
    add_files -norecurse $mem
    set_property file_type {Memory Initialization Files} [get_files $mem]
}
set_property top lut6_sequential [get_filesets sources_1]
add_files -fileset constrs_1 -norecurse [file join $root timing.xdc]
launch_runs synth_1 -jobs 4
wait_on_run synth_1
set synth_log [file join $root vivado_build_$tag "teacher_lut6_$tag.runs" synth_1 runme.log]
set fd [open $synth_log r]
set synth_text [read $fd]
close $fd
if {[string first "Synth 8-4445" $synth_text] >= 0} {
    error "Synthesis could not load a LUT6 ROM; inspect $synth_log"
}
open_run synth_1
file mkdir $outdir
report_utilization -file [file join $outdir synthesis_utilization.rpt]
report_utilization -hierarchical -file [file join $outdir synthesis_hierarchical_utilization.rpt]
set ram_file [open [file join $outdir synthesis_bram_instances.csv] w]
puts $ram_file "instance,primitive"
foreach cell [get_cells -hierarchical -filter {REF_NAME =~ RAMB36* || REF_NAME =~ RAMB18*}] {
    puts $ram_file "[get_property NAME $cell],[get_property REF_NAME $cell]"
}
close $ram_file
opt_design
place_design
file mkdir $outdir
report_utilization -file [file join $outdir placed_utilization.rpt]
route_design
file mkdir $outdir
report_route_status -file [file join $outdir route_status.rpt]
file mkdir $outdir
report_utilization -file [file join $outdir routed_utilization.rpt]
report_utilization -hierarchical -file [file join $outdir routed_hierarchical_utilization.rpt]
set routed_ram_file [open [file join $outdir routed_bram_instances.csv] w]
puts $routed_ram_file "instance,primitive"
foreach cell [get_cells -hierarchical -filter {REF_NAME =~ RAMB36* || REF_NAME =~ RAMB18*}] {
    puts $routed_ram_file "[get_property NAME $cell],[get_property REF_NAME $cell]"
}
close $routed_ram_file
report_timing_summary -file [file join $outdir routed_timing_summary.rpt]
report_power -file [file join $outdir routed_power.rpt]
write_checkpoint -force [file join $outdir routed_design.dcp]
''')
    (out / "run_sim.sh").write_text('''#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
for mem in rtl/*.mem; do cp "$mem" "$(basename "$mem")"; done
xvlog -sv rtl/lut6_sequential.sv sim/tb_lut6_sequential.sv
xelab tb_lut6_sequential -s graph_tb_snapshot
xsim graph_tb_snapshot -runall -log simulation.log
grep -q 'PREDICTIONS_PASS' simulation.log
grep -q 'ALL_TESTS_PASS' simulation.log
''')
    (out / "run_sim.sh").chmod(0o755)


def build(ir_path, out, *, random_tests=1, seed=42, clock_ns=10.0,
          debug_scores=False):
    require(random_tests >= 0 and clock_ns > 0, "Invalid test count or clock")
    ir_path, out = Path(ir_path), Path(out)
    ir = json.loads(ir_path.read_text())
    spec = compile_graph(ir)
    memories, counts = write_roms(spec, out)
    verify_rom_roundtrip(spec, out)
    render_rtl(spec, out, debug_scores=debug_scores)
    (out / "sim").mkdir(exist_ok=True)
    rng = random.Random(seed)
    n = spec["n_input"]
    patterns = [[0] * n, [1] * n,
                [int((index * 11 + (index // 17) * 3) % 19 < 9)
                 for index in range(n)]]
    patterns += [[rng.getrandbits(1) for _ in range(n)] for _ in range(random_tests)]
    scores = []
    for bits in patterns:
        expected, actual = evaluate_native(ir, bits), evaluate_program(spec, bits)
        require(actual == expected, "Teacher JSON and compressed program differ")
        scores.append(actual)
    render_tb(spec, patterns, scores, out, debug_scores=debug_scores)
    write_support(out, clock_ns)
    rom_bits = sum(width * depth for name, (width, depth) in memories.items() if name != "values")
    manifest = dict(format="teacher_lut6_graph_sv_v1", input_shape=spec["shape"],
                    input_bits=n, graph_nodes=spec["n_program"], classes=spec["n_classes"],
                    score_width_bits=spec["score_w"], class_width_bits=spec["class_w"],
                    debug_scores_output=debug_scores,
                    reference_predictions=[predicted_class(score) for score in scores],
                    distinct_truth_tables=len(spec["truth_tables"]),
                    nodes_by_active_pins=dict(sorted(counts.items())), address_width_bits=spec["addr_w"],
                    memory_shapes={name: dict(width_bits=width, depth=depth)
                                   for name, (width, depth) in memories.items()},
                    theoretical_rom_bits=rom_bits,
                    estimated_bram36_blocks=sum(_bram_estimate(*item) for item in memories.values()),
                    bram_estimate_note="Approximate RAMB36 mapping; check Vivado synthesis for actual use",
                    reference_scores=scores,
                    truth_sha256=hashlib.sha256((out / "rtl" / "lut6_truth.mem").read_bytes()).hexdigest())
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"PASS native JSON versus compressed program: {len(scores)} vectors")
    print(f"Generated {spec['n_program']:,} instructions, {len(spec['truth_tables'])} shared INITs")
    print(f"ROM {rom_bits / 1e6:.2f} Mbit; rough BRAM36 estimate "
          f"{manifest['estimated_bram36_blocks']} including value RAM; output {out.resolve()}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--random-tests", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clock-ns", type=float, default=10.0)
    parser.add_argument("--debug-scores", action="store_true",
                        help="Expose the original full-width scores_flat output for debugging")
    args = parser.parse_args()
    try:
        build(args.ir, args.out, random_tests=args.random_tests,
              seed=args.seed, clock_ns=args.clock_ns,
              debug_scores=args.debug_scores)
    except (IRValidationError, KeyError, TypeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
