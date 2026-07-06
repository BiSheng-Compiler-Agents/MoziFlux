# Performance Report

## Verification summary

- `cannsim_local_run` was executed for both the editable baseline Triton epilogue and optimized direct Triton epilogue with a sub-kernel host (`N=1,C=64,D=H=W=1, grid=1`).
- Both cannsim hosts passed correctness against a CPU reference for `sigmoid(softmax(x, dim=1))`.
- `remote_verify` passed the unit test for all optimized dispatch paths.

## Cannsim trace summary (sub-kernel)

Trace files:
- Baseline: `/tmp/cannsim_local/l2_49_baseline/cannsim_20260629221146_test_kernel/report/trace_core0.json`
- Optimized: `/tmp/cannsim_local/l2_49_opt/cannsim_20260629221341_test_kernel/report/trace_core0.json`

| Kernel | Trace events | Wall cycles | Hardware latency (cycles × 0.4 ns) | Speedup |
|---|---:|---:|---:|---:|
| Baseline Triton epilogue | 428 | 6,063 | 2,425.2 ns | 1.00x |
| Optimized Triton epilogue | 185 | 3,424 | 1,369.6 ns | 1.77x |

## Cannsim busy-cycle table by pipeline/category

Busy-cycle sums can exceed wall cycles because simulator categories overlap/stall concurrently; use wall cycles for latency and this table for bottleneck diagnosis.

| Category | Baseline busy cycles | Optimized busy cycles | Change |
|---|---:|---:|---:|
| SCALAR | 5,841 | 5,013 | -14.2% |
| SCALARLDST | 5,776 | 2,970 | -48.6% |
| PUSHQ | 1,818 | 190 | -89.5% |
| VEC | 1,157 | 673 | -41.8% |
| MTE2 | 1,042 | 1,355 | +30.0% |
| MTE3 | 722 | 1,167 | +61.6% |
| RVECEX | 369 | 223 | -39.6% |
| RVECLD | 85 | 29 | -65.9% |
| RVECST | 45 | 9 | -80.0% |
| FLOWCTRL | 9 | 9 | 0.0% |

## Top trace instructions

| Rank | Baseline instruction (cycles) | Optimized instruction (cycles) |
|---:|---|---|
| 1 | `ST_XD_XN_IMM` (3,956) | `LDP_XI_XJ_XN` (3,427) |
| 2 | `LDP_XI_XJ_XN` (3,554) | `LD_XD_XN_IMM` (1,732) |
| 3 | `LD_XD_XN_IMM` (1,820) | `ST_XD_XN_IMM` (1,238) |
| 4 | `VF` (1,798) | `DC_PRELOAD_XN_IMM` (1,031) |
| 5 | `MOV_SRC_TO_DST_ALIGNv2` (1,347) | `MOV_SRC_TO_DST_ALIGNv2` (1,025) |

## Hardware benchmark (`remote_verify`, ms)

| label | PyTorch / ACL | Baseline Triton1 | Baseline Triton2 | Optimized Triton |
|---|---:|---:|---:|---:|
| tiny_triton_path | 0.157085 | 0.217897 | 0.222244 | 0.193260 |
| small_triton_path | 0.238530 | 0.624978 | 0.611228 | 0.561445 |
| default_acl_path | 41.180523 | inf (grid preskip) | inf (grid preskip) | 41.257828 |

Default-path optimized ratio vs PyTorch/ACL reference: `41.180523 / 41.257828 = 0.9981x` (latency parity). Baseline Triton providers are intentionally `inf` on the default path because their one-row-per-program launch would exceed the Ascend grid cap and poison the context.
