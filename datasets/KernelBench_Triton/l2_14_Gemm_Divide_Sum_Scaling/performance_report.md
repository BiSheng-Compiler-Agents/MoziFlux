# Performance Report

## Cannsim setup

Sub-kernel cannsim used `grid=(1,)` with deterministic fp32 host buffers. Because the full production tile `BLOCK_M=128, BLOCK_K=64, BLOCK_N=16` generated an unsafe long trace for the padded Cube fallback, the final comparable diagnostic trace uses a bounded 16x16 tile for both baseline-vector and optimized-Cube fallback. Production speed is determined by remote hardware profiling of `profile_kernels.py`.

Trace files:
- Baseline diagnostic: `/tmp/cannsim_local/l2_14_baseline_vec_16/cannsim_20260625101248_test_kernel/report/trace_core0.json`
- Optimized diagnostic fallback: `/tmp/cannsim_local/l2_14_opt_cube_16/cannsim_20260625101432_test_kernel/report/trace_core0.json`

## Cannsim trace summary

| Kernel | Tile | wall cycles | hardware ns (`cycles*0.4`) | x_events | i_events | bottleneck |
|---|---:|---:|---:|---:|---:|---|
| Baseline vector rowwise dot | 16x16 | 3,605 | 1,442.0 | 350 | 24 | SCALAR 1,930 cyc |
| Optimized Triton Cube fallback | 16x16 | 7,492 | 2,996.8 | 1,384 | 244 | SCALARLDST 3,901 cyc |

## Pipeline utilization

| Kernel | SCALAR | SCALARLDST | MTE2 | MTE3 | VEC/RVEC | CUBE | FLOWCTRL | PUSHQ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline vector | 1,930 | 1,780 | 978 | 1,677 | VEC 964 / RVECEX 119 | 0 | 7 | 652 |
| Optimized fallback | 1,798 | 3,901 | 918 | 3,225 | VEC 172 / RVECEX 116 | 153 | 2,683 | 1,384 |

## Interpretation

The custom Triton Cube fallback is not the production path: padding GEMV from N=1 to N=16 increases scalar/local-store overhead and regresses the small diagnostic tile. The production optimization uses a fused ACL GEMV fast path for small/medium shapes where reassociation stays within 1e-3, and falls back to exact PyTorch reduction order for the large target shape where fused reassociation reached ~0.01 max-abs error.

## Hardware latency

Remote verification: `test_passed=true`, `bench_passed=true`.

| label | PyTorch / ACL ms | Baseline Triton1 ms | Baseline Triton2 ms | Optimized Triton ms | Speedup vs PyTorch / ACL |
|---|---:|---:|---:|---:|---:|
| small | 0.026292 | inf | 0.004609 | 0.005132 | 5.12x |
| medium | 0.787538 | inf | 0.017970 | 0.016015 | 49.18x |
| target | 6.406435 | inf | 0.054942* | 6.408098 | 1.00x |

\* Baseline Triton2 target is not correctness-comparable (`max_abs=0.00610352`), so its target latency is informational only.
