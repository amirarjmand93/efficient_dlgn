"""Experimental GroupSum ROM packing for the sequential LUT6 bridge.

Each sum entry stores one source address plus its inversion bit in an 18-bit
word, replacing the paired-address ROM and separate inversion ROM.
"""

import argparse
import json
from pathlib import Path

from .lut6_json_sv_bridge.lut6_json_bridge import graph_compiler as bridge


def replace_one(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"Unexpected generated RTL/Tcl: {old[:60]}")
    return text.replace(old, new)


def build(ir_path: Path, out: Path, random_tests: int = 1):
    manifest = bridge.build(ir_path, out, random_tests=random_tests,
                            debug_scores=True)
    spec = bridge.compile_graph(json.loads(ir_path.read_text()))
    addr_w = spec["addr_w"]
    width = addr_w + 1
    entries = [source | (int(inverted) << addr_w)
               for source, inverted in spec["sums"]]
    rtl_dir = out / "rtl"
    bridge._hex_lines(rtl_dir / "lut6_sum_addr.mem", entries, width)
    (rtl_dir / "lut6_sum_inv.mem").unlink()
    assert [((word & ((1 << addr_w) - 1)), bool(word >> addr_w))
            for word in (int(line, 16) for line in
                         (rtl_dir / "lut6_sum_addr.mem").read_text().splitlines())] == spec["sums"]

    rtl_path = rtl_dir / "lut6_sequential.sv"
    rtl = rtl_path.read_text()
    rtl = replace_one(rtl, "localparam integer SUM_ADDR_DEPTH=(N_SUM+1)/2;",
                      "localparam integer SUM_ADDR_DEPTH=N_SUM;")
    rtl = replace_one(rtl, "    localparam integer SUM_INV_DEPTH=(N_SUM+15)/16;\n", "")
    rtl = replace_one(rtl, "reg [2*ADDR_W-1:0] sum_pair_q;\n    reg [15:0] sum_inv_word_q;",
                      "reg [ADDR_W:0] sum_entry_q;")
    rtl = replace_one(rtl, '''(* rom_style="block" *) reg [2*ADDR_W-1:0]
        sum_addr_mem[0:SUM_ADDR_DEPTH-1];
    (* rom_style="block" *) reg [15:0] sum_inv_mem[0:SUM_INV_DEPTH-1];
    initial begin
        $readmemh("lut6_sum_addr.mem",sum_addr_mem);
        $readmemh("lut6_sum_inv.mem",sum_inv_mem);
    end
    always @(posedge clk) if(state==S_SUM_FETCH) begin
        sum_pair_q <= sum_addr_mem[sum_index >> 1];
        sum_inv_word_q <= sum_inv_mem[sum_index >> 4];
    end''', '''(* rom_style="block" *) reg [ADDR_W:0]
        sum_addr_mem[0:SUM_ADDR_DEPTH-1];
    initial $readmemh("lut6_sum_addr.mem",sum_addr_mem);
    always @(posedge clk) if(state==S_SUM_FETCH)
        sum_entry_q <= sum_addr_mem[sum_index];''')
    rtl = replace_one(rtl, '''read_addr<=sum_index[0] ? sum_pair_q[2*ADDR_W-1:ADDR_W]
                                        : sum_pair_q[ADDR_W-1:0];
                sum_inv_q<=sum_inv_word_q[sum_index & 4'hF];''',
                      '''read_addr<=sum_entry_q[ADDR_W-1:0];
                sum_inv_q<=sum_entry_q[ADDR_W];''')
    rtl_path.write_text(rtl)

    tcl_path = out / "vivado" / "implement.tcl"
    tcl = replace_one(tcl_path.read_text(),
                      "if {[llength $mem_files] != 10}",
                      "if {[llength $mem_files] != 9}")
    tcl_path.write_text(tcl)
    manifest["memory_shapes"]["sum_addr"] = {"width_bits": width,
                                              "depth": len(entries)}
    del manifest["memory_shapes"]["sum_inv"]
    old_bits = 2 * addr_w * ((len(entries) + 1) // 2) + 16 * ((len(entries) + 15) // 16)
    manifest["theoretical_rom_bits"] += width * len(entries) - old_bits
    manifest["estimated_bram36_blocks"] = sum(
        bridge._bram_estimate(item["width_bits"], item["depth"])
        for item in manifest["memory_shapes"].values())
    manifest["beta_sum_packed_inversion"] = True
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"BETA sum: {len(entries):,} entries, {width}-bit address+inversion; "
          f"ROM bit change {width * len(entries) - old_bits:+,}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--random-tests", type=int, default=1)
    args = parser.parse_args()
    build(args.ir, args.out, args.random_tests)


if __name__ == "__main__":
    main()
