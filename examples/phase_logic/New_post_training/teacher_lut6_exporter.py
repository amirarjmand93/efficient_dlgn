"""Export a discrete CompactLogic teacher as an exact packed LUT6 graph.

The input interface is encoded Boolean CHW bits. This is a graph IR, not the
student's layer-oriented hls4ml_lut6_ir_v1 format or a ready HLS project.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from torchlogix.circuit import (
    LUT6Gate, LUT6Signal, LUT6SumReduction, NativeLUT6Circuit, _eval_gate_op,
)

from examples.phase_logic.post_training.export_circuit import (
    compact_to_circuit, verification_data, verify_circuit,
)
from examples.phase_logic.post_training.load_checkpoint import load_checkpoint


def pack_lut6(circuit, *, metadata=None):
    """Greedily absorb exclusive fanout-one subgraphs with <=6 boundary bits."""
    gates = {gate.gate_id: gate for gate in circuit.gates}
    fanout = Counter()
    for gate in circuit.gates:
        fanout.update(source for source in (gate.in0, gate.in1) if source in gates)
    for reduction in circuit.sum_nodes:
        fanout.update(source for source in reduction.input_ids if source in gates)

    def boundary(members):
        # Stable pin order follows the original topological gate order.
        result = []
        for gate_id in sorted(members):
            gate = gates[gate_id]
            for source in (gate.in0, gate.in1):
                if source >= 0 and source not in members and source not in result:
                    result.append(source)
        return result

    cones = {}
    for gate in circuit.gates:
        members = {gate.gate_id}
        while True:
            candidate = None
            for source in boundary(members):
                if source in cones and fanout[source] == 1:
                    merged = members | cones[source]
                    if len(boundary(merged)) <= 6:
                        candidate = merged
                        break
            if candidate is None:
                break
            members = candidate
        if len(boundary(members)) > 6:
            raise ValueError(f"Gate {gate.gate_id} exceeds six independent inputs.")
        cones[gate.gate_id] = members

    absorbed = set()
    for members in cones.values():
        if len(members) > 1:
            # Only a later gate can absorb an earlier gate.
            absorbed.update(members - {max(members)})

    packed = []
    for gate in circuit.gates:
        if gate.gate_id in absorbed:
            continue
        members = cones[gate.gate_id]
        pins = boundary(members)
        init = 0
        for address in range(64):
            values = {source: bool((address >> index) & 1)
                      for index, source in enumerate(pins)}
            for member_id in sorted(members):
                member = gates[member_id]
                a = values[member.in0] if member.in0 >= 0 else False
                b = values[member.in1] if member.in1 >= 0 else False
                values[member_id] = _eval_gate_op(member.op, a, b)
            init |= int(values[gate.gate_id]) << address
        packed.append(LUT6Gate(
            gate.gate_id,
            tuple([LUT6Signal(source) for source in pins]
                  + [LUT6Signal(-1)] * (6 - len(pins))),
            init, layer=str(gate.node_idx), name=f"teacher_gate_{gate.gate_id}",
        ))

    reductions = []
    for output in circuit.outputs:
        reduction = next((item for item in circuit.sum_nodes if item.node_id == output), None)
        if reduction is None or reduction.tau != 1 or reduction.beta != int(reduction.beta):
            raise ValueError("Packed export requires exact integer class-count reductions.")
        reductions.append(LUT6SumReduction(
            tuple(LUT6Signal(source) for source in reduction.input_ids),
            int(reduction.beta),
        ))
    largest = max((len(item.signals) + item.offset for item in reductions), default=0)
    result = NativeLUT6Circuit(circuit.input_shape, packed, reductions,
                               metadata=metadata, score_bits=max(1, largest.bit_length()))
    return result.simplified()


def export_teacher(checkpoint, output, *, verify_samples=4, config_path=None):
    if verify_samples < 1:
        raise ValueError("--verify-samples must be positive.")
    checkpoint, output = Path(checkpoint).expanduser().resolve(), Path(output).expanduser().resolve()
    model, report = load_checkpoint(checkpoint, config_path=config_path,
                                    device="cpu", backend="torch")
    encoded, expected, score_transform = verification_data(model, samples=verify_samples)
    circuit = compact_to_circuit(model)
    verify_circuit(circuit, encoded, expected)
    before = len(circuit.gates)
    circuit.simplify(n_max=100)
    verify_circuit(circuit, encoded, expected)
    simplified = len(circuit.gates)
    encoder = model[0]
    metadata = {
        "source_checkpoint": str(checkpoint),
        "source_step": report.get("step"),
        "dataset": report["dataset"],
        "model_variant": report.get("model_variant"),
        "encoded_input_shape": list(circuit.input_shape),
        "input_interface": "pre-encoded Boolean bits in CHW flatten order",
        "input_preprocessing": ("binary MNIST image (nonzero pixels -> 1)"
                                if report["dataset"] == "mnist" else
                                "saved Heartbeat Q2 + local comparison encoder"
                                if report["dataset"] == "heartbeat" else "saved CIFAR encoder"),
        "encoder_state": {name: tensor.detach().cpu().tolist()
                          for name, tensor in encoder.state_dict().items()},
        "source_score_transform": score_transform,
        "score_semantics": "raw integer GroupSum class counts including folded constant offsets",
        "pin0_is_address_lsb": True,
        "optimization": {
            "raw_two_input_gates": before,
            "simplified_two_input_gates": simplified,
            "passes": ["constant_propagation", "duplicate_elimination",
                       "dead_gate_removal", "boolean_simplification",
                       "exclusive_fanout_lut6_cone_packing"],
        },
    }
    native = pack_lut6(circuit, metadata=metadata)
    actual = native(encoded)
    if not np.array_equal(actual, expected):
        raise ValueError("Packed LUT6 graph differs from teacher integer class counts.")
    payload = native.to_dict()
    again = NativeLUT6Circuit.from_dict(payload)
    if not np.array_equal(again(encoded), expected):
        raise ValueError("LUT6 JSON round-trip differs from teacher integer class counts.")
    output.parent.mkdir(parents=True, exist_ok=True)
    native.write_json(output)
    print(f"Saved {len(native.gates):,} packed LUT6 nodes to {output}")
    print(f"Verified {verify_samples} samples: {before:,} raw -> "
          f"{simplified:,} simplified two-input -> {len(native.gates):,} packed LUT6 nodes")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--verify-samples", type=int, default=4)
    args = parser.parse_args()
    export_teacher(args.checkpoint, args.output,
                   verify_samples=args.verify_samples, config_path=args.config)


if __name__ == "__main__":
    main()
