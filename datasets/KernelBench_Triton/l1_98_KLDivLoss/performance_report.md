# Performance Report: 98_KLDivLoss

## cannsim setup

- Baseline probe: `_kl_div_batch_sum_kernel`, `B=1`, `D=1024`, `BLOCK_SIZE=1024`, `grid=1`.
- Optimized probe: `_kl_div_row_contig_kernel`, `B=1`, `D=1024`, `BLOCK_SIZE=1024`, `grid=1`.
- Both probes compute the same probability-input KL row sum on deterministic non-zero data.

## cannsim trace summary

| Probe | trace_core0.json | wall cycles | hardware ns @0.4 ns/cyc | Bottleneck | Notes |
|---|---:|---:|---:|---|---|
| Baseline Triton row kernel | `/tmp/cannsim_local/kb98_baseline/cannsim_20260625081327_test_kernel/report/trace_core0.json` | 3,299 | 1,319.6 | SCALARLDST 2,351 cyc | strided pointer arguments after host contiguous copy |
| Optimized row-contiguous kernel | `/tmp/cannsim_local/kb98_optimized_final/cannsim_20260625083105_test_kernel/report/trace_core0.json` | 3,917 | 1,566.8 | SCALARLDST 1,916 cyc | persistent row-loop legality adds sub-kernel control overhead |

## Pipeline table

| Probe | SCALARLDST | MTE2 | VEC | PUSHQ | SCALAR | RVECEX | RVECLD | RVECST | MTE3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline row | 2,351 | 993 | 962 | 914 | 734 | 245 | 132 | 128 | 104 |
| Optimized row-contiguous | 1,916 | 999 | 987 | 835 | 672 | 232 | 123 | 135 | 105 |

The optimized sub-kernel reduces SCALARLDST, PUSHQ, SCALAR, RVECEX, and RVECLD activity, but its persistent row-loop scaffolding increases sub-kernel wall cycles. Full hardware timing is therefore the deciding metric.

## Hardware benchmark (`remote_verify`)

| label | PyTorch / ACL ms | Baseline Triton1 ms | Baseline Triton2 ms | Optimized Triton ms | Speedup vs Baseline Triton1 |
|---|---:|---:|---:|---:|---:|
| small_8x256 | 0.011939 | 0.006834 | 0.006295 | 0.002825 | 2.42x |
| medium_128x4096 | 0.021714 | 0.014414 | 0.017670 | 0.010306 | 1.40x |
| irregular_17x3000 | 0.014882 | 0.008362 | 0.012432 | 0.004144 | 2.02x |
| exact_16384x16384 | 5.946422 | 2.145809 | 1.648389 | 2.136712 | 1.00x |

Correctness: `UNIT_TEST PASS` for baseline1, baseline2, optimized, and optimized fallback on all benchmark shapes.
