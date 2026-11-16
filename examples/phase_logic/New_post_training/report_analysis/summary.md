# FPGA report comparison

Source: `/home/j3k9j/project/Phasefield_DLGN/torchlogix/examples/phase_logic/New_post_training/outputs`. Runs are compared only within the same exact FPGA part.
Missing reports are shown as —. Resource tables keep synthesis, placement, and routing separate.

## xc7z030ffg676-2

### Synthesis utilization

| Run | LUTs | Flip flops | BRAM tiles | DSPs | I/O pins |
| --- | ---: | ---: | ---: | ---: | ---: |
| [cifar100_seed_4242449222](<../outputs/cifar100_seed_4242449222/graph_sv/reports_z7030/analysis.md>) | 29,701 | 1,912 | 247.0 | 0 | 17 |

- [cifar100_seed_4242449222/graph_sv/reports_z7030](<../outputs/cifar100_seed_4242449222/graph_sv/reports_z7030/analysis.md>): no routing report; implementation status is incomplete.

## xc7z035ffg676-2

### Synthesis utilization

| Run | LUTs | Flip flops | BRAM tiles | DSPs | I/O pins |
| --- | ---: | ---: | ---: | ---: | ---: |
| [cifar100_infusion](<../outputs/cifar100_infusion/graph_sv/reports_z7035/analysis.md>) | 2,951 | 1,228 | 250.5 | 0 | 17 |
| [cifar100_seed_4242449222](<../outputs/cifar100_seed_4242449222/graph_sv/reports_z7035/analysis.md>) | 3,224 | 1,238 | 308.5 | 0 | 17 |
| [cifar10_plain_seed_424244921](<../outputs/cifar10_plain_seed_424244921/graph_sv_7035/reports_z7035/analysis.md>) | 816 | 357 | 306.5 | 0 | 149 |
| [cifar10_progressive_seed_4242449222](<../outputs/cifar10_progressive_seed_4242449222/graph_sv_7035/reports_z7035/analysis.md>) | 790 | 353 | 259.5 | 0 | 149 |
| [mnist_hybrid](<../outputs/mnist_hybrid/graph_sv/reports_z7035/analysis.md>) | 785 | 323 | 214.5 | 0 | 14 |
| [mnist_hybrid_infusion](<../outputs/mnist_hybrid_infusion/graph_sv/reports_z7035/analysis.md>) | 806 | 325 | 203.0 | 0 | 14 |

### Placed utilization

| Run | LUTs | Flip flops | BRAM tiles | DSPs | I/O pins |
| --- | ---: | ---: | ---: | ---: | ---: |
| [cifar100_infusion](<../outputs/cifar100_infusion/graph_sv/reports_z7035/analysis.md>) | 2,940 | 1,231 | 250.5 | 0 | 17 |
| [cifar100_seed_4242449222](<../outputs/cifar100_seed_4242449222/graph_sv/reports_z7035/analysis.md>) | 3,047 | 1,241 | 308.5 | 0 | 17 |
| [cifar10_plain_seed_424244921](<../outputs/cifar10_plain_seed_424244921/graph_sv_7035/reports_z7035/analysis.md>) | 694 | 472 | 306.5 | 0 | 149 |
| [cifar10_progressive_seed_4242449222](<../outputs/cifar10_progressive_seed_4242449222/graph_sv_7035/reports_z7035/analysis.md>) | 673 | 468 | 259.5 | 0 | 149 |
| [mnist_hybrid](<../outputs/mnist_hybrid/graph_sv/reports_z7035/analysis.md>) | 698 | 327 | 214.5 | 0 | 14 |
| [mnist_hybrid_infusion](<../outputs/mnist_hybrid_infusion/graph_sv/reports_z7035/analysis.md>) | 722 | 331 | 203.0 | 0 | 14 |

### Routed utilization

| Run | LUTs | Flip flops | BRAM tiles | DSPs | I/O pins |
| --- | ---: | ---: | ---: | ---: | ---: |
| [cifar100_infusion](<../outputs/cifar100_infusion/graph_sv/reports_z7035/analysis.md>) | 2,940 | 1,231 | 250.5 | 0 | 17 |
| [cifar100_seed_4242449222](<../outputs/cifar100_seed_4242449222/graph_sv/reports_z7035/analysis.md>) | 3,047 | 1,241 | 308.5 | 0 | 17 |
| [cifar10_plain_seed_424244921](<../outputs/cifar10_plain_seed_424244921/graph_sv_7035/reports_z7035/analysis.md>) | 694 | 472 | 306.5 | 0 | 149 |
| [cifar10_progressive_seed_4242449222](<../outputs/cifar10_progressive_seed_4242449222/graph_sv_7035/reports_z7035/analysis.md>) | 673 | 468 | 259.5 | 0 | 149 |
| [mnist_hybrid](<../outputs/mnist_hybrid/graph_sv/reports_z7035/analysis.md>) | 698 | 327 | 214.5 | 0 | 14 |
| [mnist_hybrid_infusion](<../outputs/mnist_hybrid_infusion/graph_sv/reports_z7035/analysis.md>) | 722 | 331 | 203.0 | 0 | 14 |

### Routed visual comparison

Bars scale to the largest measured value for this FPGA part.

| Run | LUTs | BRAM tiles | Critical delay |
| --- | --- | --- | --- |
| [cifar100_infusion](<../outputs/cifar100_infusion/graph_sv/reports_z7035/analysis.md>) | ███████████████████░ 2,940 | ████████████████░░░░ 250.5 | ████████████████████ 10.887 ns |
| [cifar100_seed_4242449222](<../outputs/cifar100_seed_4242449222/graph_sv/reports_z7035/analysis.md>) | ████████████████████ 3,047 | ████████████████████ 308.5 | ███████████████████░ 10.465 ns |
| [cifar10_plain_seed_424244921](<../outputs/cifar10_plain_seed_424244921/graph_sv_7035/reports_z7035/analysis.md>) | █████░░░░░░░░░░░░░░░ 694 | ████████████████████ 306.5 | ██████████████░░░░░░ 7.792 ns |
| [cifar10_progressive_seed_4242449222](<../outputs/cifar10_progressive_seed_4242449222/graph_sv_7035/reports_z7035/analysis.md>) | ████░░░░░░░░░░░░░░░░ 673 | █████████████████░░░ 259.5 | █████████████░░░░░░░ 7.105 ns |
| [mnist_hybrid](<../outputs/mnist_hybrid/graph_sv/reports_z7035/analysis.md>) | █████░░░░░░░░░░░░░░░ 698 | ██████████████░░░░░░ 214.5 | ██████████████░░░░░░ 7.681 ns |
| [mnist_hybrid_infusion](<../outputs/mnist_hybrid_infusion/graph_sv/reports_z7035/analysis.md>) | █████░░░░░░░░░░░░░░░ 722 | █████████████░░░░░░░ 203.0 | ███████████████░░░░░ 8.226 ns |

### Routed timing, routing, and power

| Run | Clock ns | Critical delay ns | Logic ns | Routing ns | Route errors | Power W | Power confidence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| [cifar100_infusion](<../outputs/cifar100_infusion/graph_sv/reports_z7035/analysis.md>) | 10.000 | 10.887 | 4.314 | 6.573 | 0 | 0.247 | Low |
| [cifar100_seed_4242449222](<../outputs/cifar100_seed_4242449222/graph_sv/reports_z7035/analysis.md>) | 10.000 | 10.465 | 3.928 | 6.537 | 0 | 0.259 | Low |
| [cifar10_plain_seed_424244921](<../outputs/cifar10_plain_seed_424244921/graph_sv_7035/reports_z7035/analysis.md>) | 10.000 | 7.792 | 2.948 | 4.844 | 0 | — | — |
| [cifar10_progressive_seed_4242449222](<../outputs/cifar10_progressive_seed_4242449222/graph_sv_7035/reports_z7035/analysis.md>) | 10.000 | 7.105 | 0.302 | 6.803 | 0 | 0.247 | Low |
| [mnist_hybrid](<../outputs/mnist_hybrid/graph_sv/reports_z7035/analysis.md>) | 10.000 | 7.681 | 0.333 | 7.348 | 0 | 0.239 | Low |
| [mnist_hybrid_infusion](<../outputs/mnist_hybrid_infusion/graph_sv/reports_z7035/analysis.md>) | 10.000 | 8.226 | 0.330 | 7.896 | 0 | 0.238 | Low |

Critical delay is the data path delay of Vivado's worst setup path, not the complete clock period with skew and uncertainty.
Power estimates depend on activity assumptions; read each power report's confidence level.
I/O input and output delays were not constrained, so timing results cover the constrained paths only.
