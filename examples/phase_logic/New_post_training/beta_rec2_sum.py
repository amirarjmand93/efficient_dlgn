"""Combine the isolated rec2 and GroupSum BRAM packing experiments."""

import argparse
import json
from pathlib import Path

from . import beta_rec2, beta_sum
from .lut6_json_sv_bridge.lut6_json_bridge import graph_compiler as bridge


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--random-tests", type=int, default=1)
    args = parser.parse_args()
    manifest = beta_sum.build(args.ir, args.out, args.random_tests)
    spec = bridge.compile_graph(json.loads(args.ir.read_text()))
    beta_rec2.apply(spec, args.out, manifest)


if __name__ == "__main__":
    main()
