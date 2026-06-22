# Performance Report

## Cannsim setup

- Baseline sub-kernel: `_bias_scale_sigmoid_kernel`, 1 program, 1,024 fp32 elements, `HW=256`, `C=32`, `BLOCK_SIZE=1024` to exercise the baseline per-element channel calculation.
- Optimized sub-kernel: `_bias_scale_sigmoid_row_direct`, 1 program, 1,024 fp32 elements, `HW=1024`, `C=32`, `BLOCK_HW=1024` to exercise the row-tiled contiguous path.
- Baseline `cannsim_local_run` hit the known instr.bin quiet-time guard after the host launch; a manual `cannsim report -e . -o report -n 0` produced `trace_core0.json` successfully.
- Cycle-to-time conversion: `0.4 ns/cycle`.

## Cannsim trace comparison

| Kernel | wall_cycles | hw_time_us | x_events | i_events | Bottleneck |
|---|---:|---:|---:|---:|---|
| Baseline flat epilogue | 27,551 | 11.020 | 31,485 | 1,854 | SCALARLDST 26,362 busy cycles |
| Optimized row-tiled epilogue | 3,685 | 1.474 | 241 | 10 | SCALARLDST 2,112 busy cycles |

## Pipeline utilization

| Kernel | Pipeline | ops | busy_cyc | lane_sum | Note |
|---|---|---:|---:|---:|---|
| Baseline | SCALARLDST | 3,683 | 26,362 | 73,272 | bottleneck |
| Baseline | SCALAR | 27,777 | 25,814 | 114,714 | per-element index arithmetic |
| Baseline | MTE2 | 2 | 972 | 973 | input DMA |
| Baseline | PUSHQ | 4 | 443 | 443 | dispatch |
| Optimized | SCALARLDST | 3 | 2,112 | 2,740 | bottleneck |
| Optimized | SCALAR | 96 | 1,788 | 3,996 | scalar work mostly launch/setup |
| Optimized | MTE3 | 2 | 1,007 | 1,007 | output DMA |
| Optimized | MTE2 | 3 | 974 | 1,024 | input DMA |
| Optimized | VEC | 1 | 964 | 964 | waits on MTE2 |

## Top instruction deltas

| Kernel | Instruction | Pipe | Count | total_cyc | avg_cyc | Interpretation |
|---|---|---|---:|---:|---:|---|
| Baseline | ST_XD_XN_IMM | SCALARLDST | 1,840 | 63,476 | 34 | scalar spill/store storm from flat indexing |
| Baseline | SHR | SCALAR | 5,533 | 22,132 | 4 | division lowering/index math |
| Baseline | ADD / SIGNEXT / AND | SCALAR | 3,691-3,693 | ~14,764 each | 4 | address/index arithmetic |
| Optimized | LD_XD_XN | SCALARLDST | 2 | 1,523 | 762 | setup scalar loads |
| Optimized | RV_VDIV | RVECEX | 16 | 272 | 17 | sigmoid reciprocal/divide |
| Optimized | RV_VMULS | RVECEX | 32 | 256 | 8 | scale/multiply |

## Hardware benchmark latency

`remote_verify` passed correctness (`UNIT_TEST PASS`). Benchmark timings:

| label | PyTorch / ACL (ms) | Baseline Triton1 (ms) | Baseline Triton2 (ms) | Optimized Triton (ms) | Optimized vs ACL |
|---|---:|---:|---:|---:|---:|
| small_direct | 0.041847 | inf | inf | 0.050287 | 0.832x |
| medium_direct | 0.081176 | inf | inf | 0.075196 | 1.080x |
| exact_persistent | 7.418134 | inf | inf | 5.103996 | 1.453x |

Baseline Triton benchmark columns are `inf` by design: baseline1 autotune contains grid-risky configs and baseline2 is read-only/reference-mismatching on small/medium and grid-guarded on the exact target. Correctness still executes baseline1 where legal and requires optimized PASS on every shape.
