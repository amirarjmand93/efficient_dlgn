"""Copy HLS Verilog and distribute the dense-output FSM enable by bank.

Run after HLS csynth_design. The original HLS RTL is never changed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil


_ENABLE_BLOCK = re.compile(
    r"(?m)^always @ \(posedge ap_clk\) begin\n"
    r"    if \(\(1'b1 == ap_CS_fsm_state1\)\) begin\n"
    r"(?P<body>(?:        [^\n]*\n)+)"
    r"    end\nend$"
)
_ASSIGNMENT = re.compile(r"^        (?P<target>\w+) <= (?P<source>\w+);$", re.MULTILINE)
_DECLARATION = re.compile(r"(?m)^reg\s+(?:\[(?P<high>\d+):0\]\s+)?(?P<name>\w+);$")


def localize_top(text: str):
    """Return modified RTL and a narrow audit of the rewritten CE loads."""
    if "local_dense_ce_0" in text:
        raise ValueError("The HLS top already contains local dense enables.")
    if not re.search(r"(?m)^reg\s+\[2:0\] ap_NS_fsm;$", text):
        raise ValueError("Expected the three-state HLS FSM next-state register.")
    if not re.search(r"(?m)^#0 ap_CS_fsm = 3'd1;$", text):
        raise ValueError("Expected power-on FSM state 1.")
    if not re.search(
        r"ap_CS_fsm <= ap_ST_fsm_state1;\s*end else begin\s*"
        r"ap_CS_fsm <= ap_NS_fsm;", text,
    ):
        raise ValueError("Expected synchronous FSM reset/next-state behavior.")
    matches = list(_ENABLE_BLOCK.finditer(text))
    if len(matches) != 1:
        raise ValueError(f"Expected one dense-output enable block; found {len(matches)}.")
    block = matches[0]
    lines = block.group("body").splitlines(keepends=True)
    assignments = []
    for line in lines:
        match = _ASSIGNMENT.fullmatch(line.rstrip("\n"))
        if match is None:
            raise ValueError(f"Unexpected statement in dense enable block: {line.strip()}")
        assignments.append((match["target"], match["source"]))
    if len(assignments) < 2 or not all(
        target.startswith("layer5_out") or target.startswith("trunc_ln44_reg")
        for target, _ in assignments
    ):
        raise ValueError("Dense enable block has unexpected target registers.")

    declarations = {
        match["name"]: int(match["high"]) + 1 if match["high"] else 1
        for match in _DECLARATION.finditer(text)
    }
    if len({target for target, _ in assignments}) != len(assignments):
        raise ValueError("Dense enable block assigns a register more than once.")
    if any(target not in declarations for target, _ in assignments):
        raise ValueError("Dense enable target has no matching register declaration.")
    total_bits = sum(declarations[target] for target, _ in assignments)
    if total_bits != 8800 or max(declarations[target] for target, _ in assignments) > 64:
        raise ValueError(f"Expected 8,800 bits in banks of at most 64; found {total_bits}.")

    # HLS emits bank names in lexical order. Number replicas by the Dense2
    # return index to keep the physical groups in numerical class order.
    by_index = {}
    for target, source in assignments:
        if target.startswith("layer5_out"):
            match = re.search(r"_ap_return_(\d+)$", source)
            if match is None:
                raise ValueError(f"Cannot identify output bank for {target}.")
            index = int(match.group(1))
        else:
            index = 137
        if index in by_index:
            raise ValueError(f"Duplicate dense-output bank {index}.")
        by_index[index] = target
    if set(by_index) != set(range(len(assignments))):
        raise ValueError("Dense-output bank indexes are not contiguous.")
    target_index = {target: index for index, target in by_index.items()}

    clones = "\n".join(
        f'(* DONT_TOUCH = "true", KEEP = "true" *) reg local_dense_ce_{index};'
        for index in range(len(assignments))
    )
    clones += "\n\n"
    for index in range(len(assignments)):
        clones += (
            f"initial local_dense_ce_{index} = 1'b1;\n"
            f"always @(posedge ap_clk) begin\n"
            f"    if (ap_rst == 1'b1) local_dense_ce_{index} <= 1'b1;\n"
            f"    else local_dense_ce_{index} <= ap_NS_fsm[0];\n"
            f"end\n"
        )
    clones += "\n"
    declaration_anchor = "reg   [2:0] ap_NS_fsm;\n"
    if text.count(declaration_anchor) != 1:
        raise ValueError("Cannot insert local CE declarations at the FSM state.")
    text = text.replace(declaration_anchor, declaration_anchor + clones, 1)

    # Re-find after insertion; keep each data assignment and its RHS unchanged.
    block = _ENABLE_BLOCK.search(text)
    if block is None:
        raise ValueError("Dense enable block disappeared during insertion.")
    replacement = "\n".join(
        f"always @(posedge ap_clk) begin\n"
        f"    if (local_dense_ce_{target_index[target]}) {target} <= {source};\n"
        f"end"
        for target, source in assignments
    )
    text = text[:block.start()] + replacement + text[block.end():]
    return text, {
        "dense_output_bits": total_bits,
        "original_ce_load_registers": len(assignments),
        "replica_ff_added": len(assignments),
        "largest_local_ce_load_bits": max(declarations[target] for target, _ in assignments),
        "fsm_next_state": "ap_NS_fsm[0]",
        "fsm_reset_and_init": 1,
    }


def make_local_rtl(project_dir: Path, output_dir: Path):
    project_dir = project_dir.resolve()
    source_dir = project_dir / "lut6_student_prj/solution1/syn/verilog"
    original = source_dir / "lut6_student.v"
    wrapper = project_dir / "lut6_student_banked.v"
    if not original.is_file() or not wrapper.is_file():
        raise FileNotFoundError("HLS Verilog and banked input wrapper are required.")
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    source = original.read_text()
    patched, audit = localize_top(source)
    files = sorted(source_dir.glob("*.v"))
    if len(files) < 2:
        raise ValueError("Expected HLS top plus its Verilog submodules.")
    output_dir.mkdir(parents=True)
    for path in files:
        shutil.copy2(path, output_dir / path.name)
    (output_dir / original.name).write_text(patched)
    shutil.copy2(wrapper, output_dir / wrapper.name)
    audit.update(
        source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        modified_sha256=hashlib.sha256(patched.encode()).hexdigest(),
        verilog_files=len(files) + 1,
        top="lut6_student_banked",
    )
    (output_dir / "local_ce_report.json").write_text(json.dumps(audit, indent=2) + "\n")
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    audit = make_local_rtl(args.project_dir, args.output_dir)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
