# Performance Report: l2_29_Matmul_Mish_Mish

## Cannsim trace comparison

Baseline trace: `/tmp/cannsim_local/l2_29_baseline/cannsim_20260625135043_test_kernel/report/trace_core0.json`
Optimized fallback trace: `/tmp/cannsim_local/l2_29_opt_dot64_v2/cannsim_20260625135745_test_kernel/report/trace_core0.json`

| Kernel | Simulated tile | Outputs | wall_cycles | cycles/output | Bottleneck | Notes |
|---|---:|---:|---:|---:|---|---|
| Baseline rowwise vector matmul | M=1, N=32, K=64 | 32 | 5,970 | 186.56 | MTE2 3,202 cyc | Vector FMA/reduce; no Cube use |
| Optimized Triton fallback | M=64, N=64, K=64 | 4,096 | 6,157 | 1.50 | FLOWCTRL 3,121 cyc | Cube `tl.dot`; fused Mish twice |

Normalized cannsim result: 186.56 → 1.50 cycles/output, about **124× less traced work per output element** for the fallback tile.

### Pipeline table

| Kernel | MTE2 | VEC/RVECEX | CUBE/FIXP | MTE3 | SCALAR/SCALARLDST | PUSHQ/FLOWCTRL |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 3,202 | VEC 3,147 / RVECEX 802 | 0 | 2,296 | 2,707 / 2,516 | 1,777 / 7 |
| Optimized fallback | 1,407 | RVECEX 1,606 | CUBE 1,773 / FIXP 1,883 | 2,424 | 1,049 / 1,248 | 2,799 / 3,121 |

### Top instructions

| Kernel | Top instruction | Count | Total cycles | Rationale |
|---|---|---:|---:|---|
| Baseline | `ST_XD_XN_IMM` | 49 | 22,294 | Scalar load/store overhead from rowwise vector reduction |
| Baseline | `WAIT_FLAG_MTE2` | 5 | 8,988 | Vector waiting on GM loads |
| Optimized fallback | `SET_INTRA_BLOCKI` | 2 | 3,766 | Cube/vector synchronization for fused dot + epilogue |
| Optimized fallback | `RV_VLDI` | 384 | 3,683 | Vector epilogue loads for Mish twice |

## Hardware benchmark

Measured by `remote_verify(run_test=True, run_bench=True)` on Ascend NPU. `Baseline Triton2` is preserved as a parser-visible read-only reference column and skipped by policy; `Baseline Triton1` target is skipped because its launch grid exceeds the 65,535 Ascend cap.

| label | PyTorch / ACL ms | Baseline Triton1 ms | Baseline Triton2 ms | Optimized Triton ms | Speedup vs Baseline1 |
|---|---:|---:|---:|---:|---:|
| small | 0.021892 | 0.056353 | inf | 0.021928 | 2.57× |
| medium | 0.060687 | 2.565578 | inf | 0.061053 | 42.02× |
| target | 7.025122 | inf | inf | 7.023730 | N/A |

Optimized target hardware latency: **7.023730 ms**. Correctness: `UNIT_TEST PASS`; optimized production and Triton fallback both passed.
