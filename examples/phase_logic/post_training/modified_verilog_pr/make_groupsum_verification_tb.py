#!/usr/bin/env python3
"""Generate a Verilog self-checking testbench from Torchlogix verification_vectors.npz.

Usage:
  python make_groupsum_verification_tb.py --ir circuit.json \
      --vectors verification_vectors.npz --output hierarchical/tb_counts.v

For an installed Icarus Verilog toolchain:
  iverilog -g2012 -s tb_counts -o sim.vvp hierarchical/circuit.v hierarchical/tb_counts.v
  vvp sim.vvp
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np


def generate(ir: dict, encoded: np.ndarray, expected: np.ndarray) -> str:
    n_inputs = int(ir['n_inputs'])
    output_ids = ir['outputs']
    sums = {int(sr['node_id']): sr for sr in ir['sum_nodes']}
    if any(int(i) not in sums for i in output_ids):
        raise ValueError('Expected GroupSum-only outputs.')
    width = 8 if max(len(sums[int(i)]['input_ids']) + int(sums[int(i)].get('beta', 0))
                     for i in output_ids) <= 255 else 16
    # Torchlogix uses uint8/16/32/64 based on the maximum output count.
    max_value = max(len(sums[int(i)]['input_ids']) + int(sums[int(i)].get('beta', 0))
                    for i in output_ids)
    if max_value > 65535:
        width = 32 if max_value <= 2**32-1 else 64
    x = np.asarray(encoded).reshape(len(encoded), -1)
    y = np.asarray(expected).reshape(len(expected), -1)
    if x.shape[1] != n_inputs or y.shape[1] != len(output_ids) or len(x) != len(y):
        raise ValueError('Verification vectors incompatible with this IR.')
    if not np.isin(x, (0,1)).all():
        raise ValueError('Encoded data must be Boolean.')
    lines = [
        '// Self-checking RTL testbench generated from original Torchlogix export vectors.',
        'module tb_counts;',
        f'    reg [{n_inputs-1}:0] inp;',
        f'    wire [{width*len(output_ids)-1}:0] scores_flat;',
        '    circuit dut (.inp(inp), .scores_flat(scores_flat));',
        '    initial begin',
    ]
    for i, (inp, labels) in enumerate(zip(x,y)):
        # packing little endian means encoded bit zero -> inp[0]
        data = int.from_bytes(np.packbits(inp.astype(np.uint8), bitorder='little').tobytes(), 'little')
        lines.append(f"        inp = {n_inputs}'h{data:0{(n_inputs+3)//4}x};")
        lines.append('        #10;')
        for j, value in enumerate(labels):
            value = int(value)
            lines += [
                f"        if (scores_flat[{j}*{width} +: {width}] !== {width}'d{value}) begin",
                f'            $display("ERROR sample {i} class {j}: actual=%0d expected={value}", '
                f'scores_flat[{j}*{width} +: {width}]);',
                '            $fatal(1);',
                '        end',
            ]
    lines += [
        f'        $display("PASS: exact class counts for {len(x)} input samples.");',
        '        $finish;',
        '    end',
        'endmodule',
    ]
    return '\n'.join(lines) + '\n'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ir', required=True, type=Path)
    p.add_argument('--vectors', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    ir = json.loads(a.ir.read_text())
    with np.load(a.vectors) as data:
        code = generate(ir, data['encoded'], data['class_counts'])
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(code)
    print('Saved', a.output)
    print('Compile with: iverilog -g2012 -s tb_counts -o sim.vvp <NEW-circuit.v> '+str(a.output))
    print('Then run: vvp sim.vvp')

if __name__ == '__main__':
    main()
