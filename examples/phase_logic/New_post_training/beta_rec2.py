"""Experimental rec2 ROM packing for the sequential LUT6 bridge.

Run separately from the main exporter/compiler. Two-input LUT truth tables are
stored as four bits in their record instead of a shared truth-table index.
"""

import argparse
import json
from pathlib import Path

from .lut6_json_sv_bridge.lut6_json_bridge import graph_compiler as bridge


def build(ir_path: Path, out: Path, random_tests: int = 1, debug_scores: bool = False):
    manifest = bridge.build(ir_path, out, random_tests=random_tests,
                            debug_scores=debug_scores)
    spec = bridge.compile_graph(json.loads(ir_path.read_text()))
    apply(spec, out, manifest)
    return manifest


def apply(spec, out: Path, manifest):
    addr_w = spec["addr_w"]
    old_w = 2 * addr_w + spec["id_w"]
    new_w = 2 * addr_w + 4
    rec2_items = [(pins, truth_id) for pins, truth_id in spec["instructions"] if len(pins) == 2]
    words = []
    for pins, truth_id in rec2_items:
        truth = spec["truth_tables"][truth_id]
        four = truth & 15
        if any((truth >> i) & 1 != (four >> (i & 3)) & 1 for i in range(64)):
            raise ValueError("A two-input truth table depends on an unused input")
        words.append((four << (2 * addr_w)) | pins[0] | (pins[1] << addr_w))
    rec2 = out / "rtl" / "lut6_rec2.mem"
    bridge._hex_lines(rec2, words, new_w)
    rtl_path = out / "rtl" / "lut6_sequential.sv"
    rtl = rtl_path.read_text()
    old = f"reg [{old_w - 1}:0] rec2_mem[0:{len(words) - 1}];"
    new = f"reg [{new_w - 1}:0] rec2_mem[0:{len(words) - 1}];"
    if rtl.count(old) != 1 or rtl.count(f"reg [{old_w - 1}:0] rec2_q;") != 1:
        raise ValueError("Unexpected rec2 RTL layout")
    rtl = rtl.replace(old, new).replace(f"reg [{old_w - 1}:0] rec2_q;",
                                      f"reg [{new_w - 1}:0] rec2_q;")
    select_old = f"rec_word = {{{{(REC_MAX_W-{old_w}){{1'b0}}}},rec2_q}};"
    select_new = f"rec_word = {{{{(REC_MAX_W-{new_w}){{1'b0}}}},rec2_q}};"
    if rtl.count(select_old) != 1:
        raise ValueError("Unexpected record selector")
    rtl = rtl.replace(select_old, select_new)
    write_bit = "state==S_LOAD ? input_bit : truth_q[lut_addr];"
    if rtl.count(write_bit) != 1:
        raise ValueError("Unexpected gate output bit selection")
    rtl = rtl.replace(write_bit,
                      f"state==S_LOAD ? input_bit : (kind_q==3'd1 ? "
                      f"rec2_q[{2*addr_w} + lut_addr[1:0]] : truth_q[lut_addr]);")
    rtl_path.write_text(rtl)
    for index, word in enumerate(int(line, 16) for line in rec2.read_text().splitlines()):
        # The generator above and ROM readback must agree on every record.
        expected_pins, expected_id = rec2_items[index]
        assert (word & ((1 << addr_w) - 1), (word >> addr_w) & ((1 << addr_w) - 1)) == expected_pins
        assert (word >> (2 * addr_w)) == (spec["truth_tables"][expected_id] & 15)
    manifest["memory_shapes"]["rec2"]["width_bits"] = new_w
    manifest["theoretical_rom_bits"] -= (old_w - new_w) * len(words)
    manifest["estimated_bram36_blocks"] = sum(
        bridge._bram_estimate(item["width_bits"], item["depth"])
        for item in manifest["memory_shapes"].values())
    manifest["beta_rec2_inline_truth"] = True
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"BETA rec2: {len(words):,} records, {old_w} -> {new_w} bits; "
          f"saved {(old_w-new_w)*len(words):,} ROM bits")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--random-tests", type=int, default=1)
    parser.add_argument("--debug-scores", action="store_true")
    args = parser.parse_args()
    build(args.ir, args.out, args.random_tests, args.debug_scores)


if __name__ == "__main__":
    main()
