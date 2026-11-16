# Vivado report analysis

From the repository root, run:

```bash
python -m examples.phase_logic.New_post_training.report_analysis.generate
```

The script reads every `reports_*` folder under `New_post_training/outputs`.
It writes `analysis.md` beside each folder's Vivado reports and writes a
combined `summary.md` in this directory. The summary groups results by exact
FPGA part and compares synthesis, placement, and routed resource use separately.
It also compares critical setup data path delay, routing errors, and Vivado
power estimates. The summary includes bars for routed LUTs, BRAM tiles, and
critical delay, scaled within each exact FPGA part. Each run's `analysis.md`
also shows LUT and BRAM bars across implementation stages. The reported delay
is the data path of Vivado's worst setup path; the clock period remains visible
for context.
Missing reports are marked as unavailable; a synthesis-only run is not reported
as routed. Power estimates retain Vivado's confidence level.

Use `--outputs PATH` and `--summary PATH` to change the input or summary path.
The script only reads Vivado text reports and `manifest.json`; it does not run
Vivado or load checkpoints.
