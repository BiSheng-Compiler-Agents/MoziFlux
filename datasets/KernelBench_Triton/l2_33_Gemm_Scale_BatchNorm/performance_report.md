# Performance Report

## Verification Summary

- `cannsim_local_run` baseline: PASS, trace generated at `/tmp/cannsim_local/l2_33_gemm_scale_bn_baseline/cannsim_20260629173506_test_kernel/report/trace_core0.json`.
- `cannsim_local_run` optimized: PASS, trace generated at `/tmp/cannsim_local/l2_33_gemm_scale_bn_optimized/cannsim_20260629173716_test_kernel/report/trace_core0.json`.
- Remote hardware verification: `test_passed=true`, `bench_passed=true`.
- Read-only `base_33_Gemm_Scale_BatchNorm.py` comparison provider is reported as unavailable on hardware because it hits Triton-Ascend UB overflow during MLIR lowering; it was not modified.

## Cannsim Trace Comparison

Sub-kernel simulation used one program (`grid=1`) with one K tile (`K=32`). The optimized trace processes a 128×128 tile, while the baseline trace processes a 64×64 tile; normalized per-output metrics are therefore the comparable figures.

| Metric | Baseline 64×64×32 | Optimized 128×128×32 | Normalized Result |
|---|---:|---:|---:|
| Output elements per tile | 4,096 | 16,384 | 4.00× work/tile |
| wall_cycles | 8,816 | 13,167 | 1.49× cycles for 4× work |
| hardware time (`cycles × 0.4 ns`) | 3,526.4 ns | 5,266.8 ns | 2.68× lower ns/output |
| cycles/output element | 2.1523 | 0.8036 | 2.68× improvement |
| ns/output element | 0.8609 | 0.3215 | 2.68× improvement |
| x_events | 2,331 | 3,974 | 1.70× events for 4× work |
| i_events | 111 | 132 | 1.19× events for 4× work |

## Pipeline Utilization (busy cycles)

| Pipeline | Baseline busy_cyc | Optimized busy_cyc | Notes |
|---|---:|---:|---|
| MTE3 | 5,180 | 7,438 | Still the bottleneck; optimized writes 4× larger C tile. |
| FLOWCTRL | 4,810 | 5,821 | Only 1.21× higher for 4× work after `tl.range` and 1D scheduling. |
| PUSHQ | 3,935 | 5,722 | Dispatch pressure grows sublinearly with tile work. |
| MTE2 | 3,372 | 1,836 | Improved B-tile contiguity lowers GM/L1 input movement pressure. |
| VEC | 2,665 | 1,128 | Less vector-side overhead around dot accumulation. |
| SCALAR | 2,552 | 2,427 | Scalar busy cycles slightly lower despite larger tile. |
| CUBE | 673 | 2,280 | More useful Cube work per program. |
| FIXP | 845 | 2,844 | Larger Cube result movement. |
| RVECST | 2,433 | 4,915 | More stores for 4× output elements. |
| RVECLD | 1,746 | 3,689 | More local reads for 4× output elements. |
| RVECEX | 197 | 295 | Sublinear vector execution increase. |

## Top Trace Instructions

| Rank | Baseline top instruction | Baseline total_cyc | Optimized top instruction | Optimized total_cyc |
|---:|---|---:|---|---:|
| 1 | RV_VSTI | 8,342 | RV_VSTI | 18,045 |
| 2 | MOV_SPR_XN | 7,911 | RV_VLDI | 11,538 |
| 3 | ST_XD_XN_IMM | 6,511 | WAIT_FLAG_VEC | 6,191 |
| 4 | RV_VLDI | 6,365 | SET_INTRA_BLOCKI | 5,812 |
| 5 | WAIT_FLAG_VEC | 5,883 | VF | 5,710 |
| 6 | SET_INTRA_BLOCKI | 5,051 | LDP_XI_XJ_XN | 5,588 |

## Hardware Benchmark Latency

`profile_kernels.py` was run on the remote Ascend NPU through `remote_verify`.

| Shape | PyTorch / ACL (ms) | Baseline Triton1 (ms) | Baseline Triton2 (ms) | Optimized Triton (ms) | Speedup vs Baseline Triton1 |
|---|---:|---:|---:|---:|---:|
| small_K1024 | 0.409222 | 1.508298 | inf | 0.611062 | 2.47× |
| medium_K4096 | 1.973866 | 50.022953 | inf | 4.568573 | 10.95× |
| required_K8192 | 17.538818 | 733.081360 | inf | 48.622585 | 15.08× |

The optimized Triton path is substantially faster than the editable baseline Triton kernel on all measured shapes, with the largest gain on the required 1024×8192 × 8192×8192 GEMM+scale+BatchNorm workload.
