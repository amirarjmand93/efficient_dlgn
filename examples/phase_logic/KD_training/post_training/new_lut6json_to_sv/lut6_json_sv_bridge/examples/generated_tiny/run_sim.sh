#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
xvlog -sv rtl/lut6_sequential.sv sim/tb_lut6_sequential.sv
xelab tb_lut6_sequential -s bridge_tb_snapshot
xsim bridge_tb_snapshot -runall -log simulation.log
grep -q 'ALL_TESTS_PASS' simulation.log
