"""Dependency-free tests for Dense-1 -> Dense-2 locality tiling."""

import unittest

from dense_locality import localize_dense_stages


class DenseLocalityTests(unittest.TestCase):
    def setUp(self):
        self.ir = {
            "n_inputs": 4,
            "gates": [
                {"gate_id": 4, "op": "AND", "in0": 0, "in1": 1, "node_idx": 9},
                {"gate_id": 5, "op": "XOR", "in0": 2, "in1": 3, "node_idx": 9},
                {"gate_id": 6, "op": "OR", "in0": 4, "in1": 5, "node_idx": 10},
                {"gate_id": 7, "op": "AND_NOT_B", "in0": 4, "in1": 0, "node_idx": 10},
            ],
            "sum_nodes": [
                {"node_id": 8, "input_ids": [6, 7], "beta": 0.0, "tau": 1.0},
            ],
        }
        self.verilog = """module circuit (
    input wire [3:0] inp,
    output wire [7:0] scores_flat
);
    wire g4 = (inp[0] & inp[1]);
    wire g5 = (inp[2] ^ inp[3]);
    wire g6 = (g4 | g5);
    wire g7 = (g4 & ~inp[0]);
    wire [1:0] raw;
    assign raw[0] = g6;
    assign raw[1] = g7;
endmodule
"""

    def test_connected_dense_gates_share_a_preserved_tile(self):
        transformed, stats = localize_dense_stages(self.ir, self.verilog, target_size=32)
        top = transformed.partition("\nendmodule")[0]
        self.assertEqual(stats["dense1_to_dense2_edges"], 3)
        self.assertEqual(stats["cross_tile_dense1_to_dense2_edges"], 0)
        self.assertEqual(stats["equations_validated"], 4)
        self.assertIn('(* keep_hierarchy = "yes" *)', top)
        self.assertNotIn("wire g4 =", top)
        self.assertIn("wire g6;", top)
        self.assertIn("wire g4 = (tile_in[0] & tile_in[1]);", transformed)
        self.assertIn("wire g6 = (g4 | g5);", transformed)
        self.assertIn("assign raw[0] = g6;", top)

    def test_rejects_changed_gate_equation(self):
        changed = self.verilog.replace("(g4 | g5)", "(g4 & g5)")
        with self.assertRaisesRegex(ValueError, "differs between"):
            localize_dense_stages(self.ir, changed, target_size=32)


if __name__ == "__main__":
    unittest.main()
