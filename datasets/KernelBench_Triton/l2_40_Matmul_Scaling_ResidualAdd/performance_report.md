# Performance Report

## Cannsim setup

Both cannsim runs used sub-kernel hosts with `grid=(1,1,1)` and `K=1 x BLOCK_K`, as required for tractable simulation. Baseline simulated the editable baseline tile (`64x64x32`); optimized simulated the Triton fallback tile (`128x128x32`). Both hosts performed numerical checks and printed `[HOST] PASS`.

Trace files:
- Baseline: `/tmp/cannsim_local/l2_40_baseline/cannsim_20260629201320_test_kernel/report/trace_core0.json`
- Optimized: `/tmp/cannsim_local/l2_40_optimized/cannsim_20260629201516_test_kernel/report/trace_core0.json`

## Cannsim trace comparison

| Metric | Baseline Triton sub-kernel | Optimized Triton sub-kernel | Change |
|---|---:|---:|---:|
| Tile shape | 64x64x32 | 128x128x32 | 4x outputs/tile |
| Wall cycles | 7,747 | 5,489 | -29.15% |
| Outputs per tile | 4,096 | 16,384 | +4.00x |
| Cycles/output | 1.8914 | 0.3350 | 5.65x better |
| Trace x_events | 2,635 | 463 | -82.43% |
| Correctness | max_abs_err=5.96046e-07 | max_abs_err=4.76837e-07 | PASS |
| Hardware time (cycles x 0.4 ns) | 3,098.8 ns | 2,195.6 ns | -903.2 ns |

## Pipeline utilization

| Pipeline | Baseline busy cycles | Optimized busy cycles | Notes |
|---|---:|---:|---|
| MTE3 / FIXP bottleneck | MTE3 3,503 | FIXP 3,397 | optimized shifts bottleneck to Cube/FIXP output movement |
| SCALARLDST | 3,259 | 1,356 | fewer scalar load/store instructions |
| FLOWCTRL | 3,152 | 11 | `tl.range` + single 1-D tile loop removes most control-flow overhead |
| PUSHQ | 2,128 | 441 | lower instruction dispatch pressure |
| MTE2 | 2,081 | 937 | direct `[K,N]` weight layout reduces input movement overhead |
| CUBE | 674 | 2,396 | larger tile activates more matrix work per program |
| RVECST | 1,038 | 264 | in-place dot avoids vector-side accumulator traffic |
| RVECLD | 878 | 0 reported | vector local-load traffic removed from top utilization |
| RVECEX | 197 | 8 | vector execution epilogue minimized |

## Top instruction deltas

| Area | Baseline top cost | Optimized top cost | Interpretation |
|---|---|---|---|
| Scalar stores | `ST_XD_XN_IMM`: 42,418 total cycles | not in optimized top list | eliminated by constexpr/static scheduling and direct B layout |
| Flow control | `SET_INTRA_BLOCKI`: 3,659 total cycles | FLOWCTRL total 13 lane cycles | static `tl.range` loop reduced dynamic loop bookkeeping |
| Vector stores | `RV_VSTI`: 8,159 total cycles | `RV_VSTI`: 2,304 total cycles | in-place accumulation and larger tile reduce normalized vector stores |
| Cube work | CUBE busy 674 cycles | `MMAD`: 2,115 cycles | optimized tile spends more time doing useful matmul work |

## Remote hardware benchmark

Remote verification passed correctness and benchmark. Latencies are milliseconds from `profile_kernels.py`.

| label | PyTorch / ACL | Baseline Triton1 | Baseline Triton2 | Optimized Triton | Optimized vs Baseline1 |
|---|---:|---:|---:|---:|---:|
| small_acl_128x512x512 | 0.160523 | 0.216013 | 0.202427 | 0.157611 | 1.37x |
| irregular_triton_257x512x512 | 0.227856 | 0.394795 | 0.212094 | 0.382661 | 1.03x |
| required_acl_16384x4096x4096 | 391.452087 | 697.611084 | 484.247101 | 391.466370 | 1.78x |

## Correctness

All optimized dispatch paths passed remote unit tests:
- ACL aligned path: `small_acl_128x512x512` and `required_acl_16384x4096x4096`, max_abs_diff `0`.
- Triton fallback path: `irregular_triton_257x512x512`, max_abs_diff `1.19209e-06`.
