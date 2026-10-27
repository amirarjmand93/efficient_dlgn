"""Connectivity-local hierarchy for the final two Boolean gate stages.

The transform is deliberately narrow: it moves the final two stages into small
combinational modules.  Connected components of the Dense-1 -> Dense-2 graph
are never split, so every such edge remains inside one hierarchy tile.
"""
from __future__ import annotations

from collections import defaultdict
import math
import re


GATE_LINE = re.compile(r"^    wire g(\d+) = (.*);$")
OPS = {
    "CONST_FALSE": "1'b0",
    "CONST_TRUE": "1'b1",
    "WIRE": "{a}",
    "NOT": "~{a}",
    "NOT_A": "~{a}",
    "NOT_B": "~{b}",
    "AND": "({a} & {b})",
    "OR": "({a} | {b})",
    "XOR": "({a} ^ {b})",
    "NAND": "~({a} & {b})",
    "NOR": "~({a} | {b})",
    "XNOR": "~({a} ^ {b})",
    "AND_NOT_B": "({a} & ~{b})",
    "AND_NOT_A": "(~{a} & {b})",
    "OR_NOT_B": "({a} | ~{b})",
    "OR_NOT_A": "(~{a} | {b})",
}


class _DisjointSet:
    def __init__(self, values):
        self.parent = {value: value for value in values}

    def find(self, value):
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            value, self.parent[value] = self.parent[value], root
        return root

    def union(self, left, right):
        left, right = self.find(left), self.find(right)
        if left != right:
            self.parent[right] = left


def _quantile(values: list[int], fraction: float) -> int:
    if not values:
        return 0
    values = sorted(values)
    return values[min(len(values) - 1, max(0, math.ceil(fraction * len(values)) - 1))]


def analyze_dense_connectivity(ir: dict) -> dict:
    """Return the few metrics that matter for physical Dense-stage locality."""
    gates = ir.get("gates", [])
    n_inputs = int(ir["n_inputs"])
    node_indices = sorted({int(gate.get("node_idx", -1)) for gate in gates})
    if len(node_indices) < 2:
        raise ValueError("At least two Boolean gate stages are required for dense analysis.")
    dense1_index, dense2_index = node_indices[-2:]
    by_id = {int(gate["gate_id"]): gate for gate in gates}
    dense1_ids = {gate_id for gate_id, gate in by_id.items()
                  if int(gate.get("node_idx", -1)) == dense1_index}
    dense2 = [gate for gate in gates if int(gate.get("node_idx", -1)) == dense2_index]
    fanout = {gate_id: 0 for gate_id in dense1_ids}
    d1_edges = []
    d2_inputs = 0
    earlier_inputs = 0
    for destination_index, gate in enumerate(dense2):
        for key in ("in0", "in1"):
            source = int(gate[key])
            if source < 0:
                continue
            d2_inputs += 1
            if source in dense1_ids:
                fanout[source] += 1
                d1_edges.append((destination_index, source))
            else:
                earlier_inputs += 1
    fanouts = list(fanout.values())
    correlation = 0.0
    if d1_edges:
        mean_x = sum(item[0] for item in d1_edges) / len(d1_edges)
        mean_y = sum(item[1] for item in d1_edges) / len(d1_edges)
        covariance = sum((x - mean_x) * (y - mean_y) for x, y in d1_edges)
        variance_x = sum((x - mean_x) ** 2 for x, _ in d1_edges)
        variance_y = sum((y - mean_y) ** 2 for _, y in d1_edges)
        if variance_x and variance_y:
            correlation = covariance / math.sqrt(variance_x * variance_y)

    block_ratios = []
    for start in range(0, len(dense2), 128):
        sources = [
            int(gate[key])
            for gate in dense2[start:start + 128]
            for key in ("in0", "in1")
            if int(gate[key]) in dense1_ids
        ]
        if sources:
            block_ratios.append(len(set(sources)) / len(sources))

    source_stage = {}
    for gate_id, gate in by_id.items():
        source_stage[gate_id] = int(gate.get("node_idx", -1))
    sums_by_stage = defaultdict(int)
    for reduction in ir.get("sum_nodes", []):
        for source in reduction.get("input_ids", []):
            source = int(source)
            label = (f"node_{source_stage[source]}" if source in source_stage
                     else "input" if source < n_inputs else "unknown")
            sums_by_stage[label] += 1

    sets = _DisjointSet(dense1_ids | {int(gate["gate_id"]) for gate in dense2})
    for gate in dense2:
        destination = int(gate["gate_id"])
        for key in ("in0", "in1"):
            source = int(gate[key])
            if source in dense1_ids:
                sets.union(destination, source)
    component_sizes = defaultdict(int)
    for gate_id in sets.parent:
        component_sizes[sets.find(gate_id)] += 1

    return {
        "dense1_node_idx": dense1_index,
        "dense2_node_idx": dense2_index,
        "dense1_emitted_gates": len(dense1_ids),
        "dense2_emitted_gates": len(dense2),
        "dense2_input_edges": d2_inputs,
        "dense1_to_dense2_edges": len(d1_edges),
        "earlier_stage_to_dense2_edges": earlier_inputs,
        "dense1_fanout_to_dense2": {
            "unused_sources": sum(value == 0 for value in fanouts),
            "median": _quantile(fanouts, 0.5),
            "p95": _quantile(fanouts, 0.95),
            "p99": _quantile(fanouts, 0.99),
            "maximum": max(fanouts, default=0),
        },
        "source_destination_index_correlation": correlation,
        "sequential_128_gate_block_unique_source_ratio": (
            sum(block_ratios) / len(block_ratios) if block_ratios else 0.0
        ),
        "dense_bipartite_components": len(component_sizes),
        "largest_dense_bipartite_component_gates": max(component_sizes.values(), default=0),
        "groupsum_input_edges": sum(sums_by_stage.values()),
        "groupsum_sources_by_stage": dict(sorted(sums_by_stage.items())),
    }


def _reference(node_id: int, n_inputs: int) -> str:
    if node_id < 0:
        return "1'b0"
    return f"inp[{node_id}]" if node_id < n_inputs else f"g{node_id}"


def _expression(gate: dict, reference) -> str:
    try:
        template = OPS[gate["op"]]
    except KeyError as error:
        raise ValueError(f"Unsupported Boolean operation {gate['op']!r}.") from error
    return template.format(a=reference(int(gate["in0"])), b=reference(int(gate["in1"])))


def _partition(gates: list[dict], dense1_ids: set[int], dense2_ids: set[int],
               target_size: int) -> tuple[list[list[int]], int]:
    dense_ids = dense1_ids | dense2_ids
    sets = _DisjointSet(dense_ids)
    edge_count = 0
    for gate in gates:
        if int(gate["gate_id"]) not in dense2_ids:
            continue
        for key in ("in0", "in1"):
            source = int(gate[key])
            if source in dense1_ids:
                sets.union(int(gate["gate_id"]), source)
                edge_count += 1

    components = defaultdict(list)
    for gate_id in dense_ids:
        components[sets.find(gate_id)].append(gate_id)
    ordered = [sorted(component) for component in components.values()]
    # Large components get their own tile.  Smaller components use deterministic
    # first-fit packing; no graph edge is cut by this packing step.
    ordered.sort(key=lambda component: (-len(component), component[0]))
    tiles: list[list[int]] = []
    sizes: list[int] = []
    for component in ordered:
        destination = next(
            (index for index, size in enumerate(sizes)
             if size + len(component) <= target_size),
            None,
        )
        if destination is None:
            tiles.append(list(component))
            sizes.append(len(component))
        else:
            tiles[destination].extend(component)
            sizes[destination] += len(component)
    for tile in tiles:
        tile.sort()
    tiles.sort(key=lambda tile: tile[0])
    return tiles, edge_count


def localize_dense_stages(ir: dict, verilog: str, target_size: int = 512) -> tuple[str, dict]:
    """Move the last two gate stages into connectivity-local hierarchy tiles."""
    if target_size < 32:
        raise ValueError("Dense locality tile size must be at least 32 gates.")
    gates = ir.get("gates", [])
    n_inputs = int(ir["n_inputs"])
    node_indices = sorted({int(gate.get("node_idx", -1)) for gate in gates})
    if len(node_indices) < 2:
        raise ValueError("At least two Boolean gate stages are required for dense localization.")
    dense1_index, dense2_index = node_indices[-2:]
    dense1_ids = {int(gate["gate_id"]) for gate in gates
                  if int(gate.get("node_idx", -1)) == dense1_index}
    dense2_ids = {int(gate["gate_id"]) for gate in gates
                  if int(gate.get("node_idx", -1)) == dense2_index}
    if not dense1_ids or not dense2_ids:
        raise ValueError("Could not identify the final two nonempty Boolean gate stages.")
    dense_ids = dense1_ids | dense2_ids
    by_id = {int(gate["gate_id"]): gate for gate in gates}
    if len(by_id) != len(gates):
        raise ValueError("Duplicate gate IDs in circuit.json.")

    top, separator, remainder = verilog.partition("\nendmodule")
    if not separator:
        raise ValueError("Cannot find the end of the top-level circuit module.")
    lines = top.splitlines()
    equations = {}
    dense_line_indices = []
    for index, line in enumerate(lines):
        match = GATE_LINE.fullmatch(line)
        if match and int(match.group(1)) in dense_ids:
            gate_id = int(match.group(1))
            equations[gate_id] = match.group(2)
            dense_line_indices.append(index)
    if set(equations) != dense_ids:
        missing = sorted(dense_ids - set(equations))
        raise ValueError(f"Verilog is missing {len(missing)} dense gate equations.")
    for gate_id, actual in equations.items():
        expected = _expression(by_id[gate_id], lambda value: _reference(value, n_inputs))
        if actual != expected:
            raise ValueError(f"Gate g{gate_id} differs between circuit.v and circuit.json.")

    tiles, dense_edge_count = _partition(gates, dense1_ids, dense2_ids, target_size)
    tile_for = {gate_id: index for index, tile in enumerate(tiles) for gate_id in tile}
    raw_ids = {int(node) for reduction in ir.get("sum_nodes", [])
               for node in reduction.get("input_ids", [])}
    consumers = defaultdict(set)
    for gate in gates:
        destination = int(gate["gate_id"])
        for key in ("in0", "in1"):
            source = int(gate[key])
            if source >= 0:
                consumers[source].add(destination)

    tile_details = []
    top_outputs = set()
    modules = []
    instances = []
    for tile_index, tile in enumerate(tiles):
        internal = set(tile)
        external_inputs = sorted({
            int(by_id[gate_id][key])
            for gate_id in tile
            for key in ("in0", "in1")
            if int(by_id[gate_id][key]) >= 0 and int(by_id[gate_id][key]) not in internal
        })
        exposed_outputs = sorted(
            gate_id for gate_id in tile
            if gate_id in raw_ids
            or any(consumer not in internal for consumer in consumers.get(gate_id, ()))
        )
        top_outputs.update(exposed_outputs)
        external_position = {node: index for index, node in enumerate(external_inputs)}
        name = f"elar_dense_tile_{tile_index:04d}"

        ports = []
        if external_inputs:
            ports.append(f"    input wire [{len(external_inputs)-1}:0] tile_in")
        if exposed_outputs:
            ports.append(f"    output wire [{len(exposed_outputs)-1}:0] tile_out")
        module_lines = [f"module {name} (", ",\n".join(ports), ");"]

        def local_reference(node_id: int) -> str:
            if node_id < 0:
                return "1'b0"
            if node_id in internal:
                return f"g{node_id}"
            return f"tile_in[{external_position[node_id]}]"

        for gate_id in tile:
            module_lines.append(
                f"    wire g{gate_id} = {_expression(by_id[gate_id], local_reference)};"
            )
        for output_index, gate_id in enumerate(exposed_outputs):
            module_lines.append(f"    assign tile_out[{output_index}] = g{gate_id};")
        module_lines.extend(["endmodule", ""])
        modules.append("\n".join(module_lines))

        connections = []
        if external_inputs:
            values = ", ".join(
                _reference(node, n_inputs) for node in reversed(external_inputs)
            )
            connections.append(f".tile_in({{{values}}})")
        if exposed_outputs:
            values = ", ".join(f"g{gate_id}" for gate_id in reversed(exposed_outputs))
            connections.append(f".tile_out({{{values}}})")
        instances.append(
            f"    (* keep_hierarchy = \"yes\" *) {name} tile_{tile_index:04d} "
            f"({', '.join(connections)});"
        )
        tile_details.append({
            "tile": tile_index,
            "gates": len(tile),
            "dense1_gates": sum(gate_id in dense1_ids for gate_id in tile),
            "dense2_gates": sum(gate_id in dense2_ids for gate_id in tile),
            "external_inputs": len(external_inputs),
            "exposed_outputs": len(exposed_outputs),
        })

    crossing = 0
    for gate in gates:
        destination = int(gate["gate_id"])
        if destination not in dense2_ids:
            continue
        for key in ("in0", "in1"):
            source = int(gate[key])
            if source in dense1_ids and tile_for[source] != tile_for[destination]:
                crossing += 1
    if crossing:
        raise AssertionError("Dense partition unexpectedly cut Dense-1 -> Dense-2 edges.")

    insertion = min(dense_line_indices)
    dense_line_set = set(dense_line_indices)
    kept_lines = [line for index, line in enumerate(lines) if index not in dense_line_set]
    # Removal changes indices after the insertion point, so count retained lines before it.
    insertion = sum(index < insertion and index not in dense_line_set
                    for index in range(len(lines)))
    top_block = [
        "    // --- connectivity-local hierarchy for the final two Boolean stages ---",
        f"    // {len(tiles)} tiles; all {dense_edge_count} Dense-1 -> Dense-2 edges stay inside a tile.",
    ]
    top_block.extend(f"    wire g{gate_id};" for gate_id in sorted(top_outputs))
    top_block.extend(instances)
    kept_lines[insertion:insertion] = top_block

    transformed = "\n".join(kept_lines) + separator + "\n\n" + "\n".join(modules)
    if remainder.startswith("\n"):
        remainder = remainder[1:]
    transformed += remainder
    stats = {
        "dense1_node_idx": dense1_index,
        "dense2_node_idx": dense2_index,
        "dense1_gates": len(dense1_ids),
        "dense2_gates": len(dense2_ids),
        "dense1_to_dense2_edges": dense_edge_count,
        "target_tile_gates": target_size,
        "tile_count": len(tiles),
        "largest_tile_gates": max(map(len, tiles)),
        "cross_tile_dense1_to_dense2_edges": crossing,
        "total_external_tile_inputs": sum(item["external_inputs"] for item in tile_details),
        "total_exposed_tile_outputs": sum(item["exposed_outputs"] for item in tile_details),
        "keep_hierarchy": True,
        "equations_validated": len(equations),
        "tiles": tile_details,
    }
    return transformed, stats
