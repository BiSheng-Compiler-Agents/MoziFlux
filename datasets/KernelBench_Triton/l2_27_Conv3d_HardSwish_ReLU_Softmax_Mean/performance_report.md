# Performance Report

## Cannsim setup

- Simulator: `cannsim_local_run(..., gen_report=True)`, SoC `Ascend950`
- Baseline trace: `/tmp/cannsim_local/kb27_baseline/cannsim_20260625125407_test_kernel/report/trace_core0.json`
- Optimized trace: `/tmp/cannsim_local/kb27_optimized/cannsim_20260625125648_test_kernel/report/trace_core0.json`
- Sub-kernel: one post-op tile with `N=1, C=16, S=512`, grid `(1,)`
- Cycle conversion: `hardware_time_ns = cycles * 0.4`

## Trace summary

| Kernel | wall_cycles | est. hardware ns | x_events | i_events | Bottleneck |
|---|---:|---:|---:|---:|---|
| Baseline fused post-op | 15,507 | 6,202.8 | 8,318 | 179 | MTE3 / `WAIT_FLAG_VEC` |
| Optimized partial post-op | 15,376 | 6,150.4 | 8,273 | 183 | MTE3 / `WAIT_FLAG_VEC` |
| Delta | -131 (-0.84%) | -52.4 ns | -45 | +4 | same per-tile bottleneck |

## Pipeline utilization

| Pipeline | Baseline busy_cyc | Optimized busy_cyc | Delta |
|---|---:|---:|---:|
| MTE3 | 13,472 | 12,770 | -702 |
| PUSHQ | 12,950 | 12,707 | -243 |
| RVECLD | 10,166 | 10,089 | -77 |
| RVECEX | 6,567 | 6,613 | +46 |
| RVECST | 4,538 | 4,570 | +32 |
| SCALARLDST | 2,348 | 1,736 | -612 |
| SCALAR | 2,021 | 1,977 | -44 |
| MTE2 | 1,077 | 1,078 | +1 |

## Interpretation

The optimized per-tile Triton candidate has nearly the same vector softmax body as the baseline, so cannsim shows only a small per-tile reduction. Full hardware profiling showed mature ACL post-ops are faster than both custom Triton variants for the benchmarked multi-tile regimes, so production dispatch routes `n_tiles > 1` to ACL and retains the traced Triton path only for tiny single-tile coverage.

## Hardware latency

Remote verification: `UNIT_TEST PASS`, `bench_passed=true`.

| Shape | PyTorch / ACL ms | Baseline Triton1 ms | Baseline Triton2 ms | Optimized Triton ms | Speedup vs Baseline1 |
|---|---:|---:|---:|---:|---:|
| small | 0.028830 | 0.035864 | 0.033349 | 0.028806 | 1.245x |
| medium | 0.109348 | 0.146655 | 0.151047 | 0.109405 | 1.340x |
| target | 0.740930 | 0.956083 | 0.988924 | 0.740613 | 1.291x |

Optimized target latency is effectively equal to the PyTorch / ACL reference while preserving `ModelNew` compatibility and Triton coverage for the tiny single-tile path.
