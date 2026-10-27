"""Dependency-free tests for the hierarchical GroupSum Verilog transformer."""

import unittest

from generate_hierarchical_groupsum import START_MARKER, _assemble


def _fixture():
    class0 = list(range(129))
    class1 = [129]
    ir = {
        "n_inputs": 130,
        "outputs": [200, 201],
        "sum_nodes": [
            {"node_id": 200, "input_ids": class0, "beta": 0.0, "tau": 1.0},
            {"node_id": 201, "input_ids": class1, "beta": 3.0, "tau": 1.0},
        ],
    }
    assignments = "\n".join(
        f"    assign raw[{index}] = inp[{node}];"
        for index, node in enumerate(class0 + class1)
    )
    verilog = f"""module circuit (
    input wire [129:0] inp,
    output reg [15:0] scores_flat
);
    wire [129:0] raw;
{assignments}

{START_MARKER}
    integer s_0, s_1, i;
    always @(*) begin
        s_0 = 0;
        for (i = 0; i < 129; i = i + 1) s_0 = s_0 + raw[i];
        scores_flat[0*8 +: 8] = s_0;
        s_1 = 3;
        for (i = 129; i < 130; i = i + 1) s_1 = s_1 + raw[i];
        scores_flat[1*8 +: 8] = s_1;
    end
endmodule
"""
    return ir, verilog


class HierarchicalGroupSumTests(unittest.TestCase):
    def test_replaces_only_output_reduction(self):
        ir, original = _fixture()
        modified, stats = _assemble(ir, original, keep_hierarchy=False)

        original_prefix = original.partition(START_MARKER)[0]
        modified_prefix = modified.partition("    // --- hierarchical GroupSum")[0]
        self.assertEqual(
            modified_prefix,
            original_prefix.replace("output reg", "output wire", 1),
        )
        self.assertEqual(stats["total_chunk_instances"], 3)
        self.assertIn("elar_popcount128 pc_c0_0", modified)
        self.assertIn("elar_popcount128 pc_c0_1", modified)
        self.assertIn("elar_popcount128 pc_c1_0", modified)
        self.assertIn("wire [8:0] gs_c0_l1_0", modified)
        self.assertIn("gs_c1_chunk0 + 8'd3", modified)
        self.assertNotIn("always @(*)", modified)

    def test_rejects_ir_with_different_folded_constant(self):
        ir, original = _fixture()
        ir["sum_nodes"][1]["beta"] = 2.0
        with self.assertRaisesRegex(ValueError, "differs from circuit.json"):
            _assemble(ir, original, keep_hierarchy=False)

    def test_rejects_ir_with_different_raw_connection(self):
        ir, original = _fixture()
        ir["sum_nodes"][0]["input_ids"][0] = 1
        with self.assertRaisesRegex(ValueError, "RTL and JSON differ"):
            _assemble(ir, original, keep_hierarchy=False)


if __name__ == "__main__":
    unittest.main()
