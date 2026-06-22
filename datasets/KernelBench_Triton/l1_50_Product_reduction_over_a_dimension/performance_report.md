# Performance Report

## Cannsim setup
- Tool: `cannsim_local_run(gen_report=True)` on Ascend950.
- Sub-kernel: `B=1, M=64, K=64`, grid `(1,)`.
- Note: the original editable baseline dynamic-`while` kernel aborted Bisheng during cannsim compilation (`Flattener::adjustOperations`); the baseline trace below uses a compileable representative of the same 8-row streaming algorithm for trace comparison.

## Cannsim trace comparison

| Kernel | wall_cycles | hardware time (ns, cycles×0.4) | x_events | i_events | Bottleneck |
|---|---:|---:|---:|---:|---|
| Baseline representative | 8,823 | 3,529.2 | 2,216 | 179 | MTE2, 6,108 busy cycles |
| Optimized (`BLOCK_M=64`, `BLOCK_K=64`) | 5,521 | 2,208.4 | 894 | 45 | SCALARLDST, 2,591 busy cycles |

Speedup from cannsim wall cycles: **1.60×** (`8823 / 5521`).

## Pipeline table

| Pipeline | Baseline busy cycles | Optimized busy cycles | Delta |
|---|---:|---:|---:|
| MTE2 | 6,108 | 1,030 | -83.1% |
| VEC | 5,494 | 1,021 | -81.4% |
| SCALARLDST | 3,556 | 2,591 | -27.1% |
| SCALAR | 3,027 | 943 | -68.8% |
| MTE3 | 1,036 | 2,217 | +114.0% |
| PUSHQ | 893 | 1,154 | +29.2% |
| RVECEX | 202 | 484 | +139.6% |
| RVECST | 130 | 348 | +167.7% |
| RVECLD | 122 | 265 | +117.2% |
| FLOWCTRL | 7 | 7 | 0.0% |

## Top instruction changes

| Instruction | Baseline | Optimized | Notes |
|---|---:|---:|---|
| `MOV_SRC_TO_DST_ALIGNv2` | 38,661 total cycles / 64 count | 1,021 total cycles / 1 count | Block load reduces repeated row-stream MTE movement. |
| `MOV_SPR_XN` | 33,911 / 65 | 966 / 2 | Fewer address setup events. |
| `WAIT_FLAG_MTE2` | 5,492 / 8 | 1,021 / 1 | Less MTE wait from fewer loop iterations. |
| `JUMPC` | 149 | 18 | `BLOCK_M=64` cuts loop control. |

## Hardware latency

Measured by `remote_verify(run_test=True, run_bench=True)` on physical Ascend NPU:

| Shape | PyTorch / ACL (ms) | Baseline Triton1 | Baseline Triton2 | Optimized Triton (ms) | Optimized vs PyTorch |
|---|---:|---:|---:|---:|---:|
| small `(4,64,64)` | 0.003217 | inf (pre-skipped) | inf (pre-skipped) | 0.002277 | 1.41× faster |
| target `(16,256,256)` | 0.013215 | inf (pre-skipped) | inf (pre-skipped) | 0.010224 | 1.29× faster |
| oddK `(7,129,193)` | 0.005293 | inf (pre-skipped) | inf (pre-skipped) | 0.005481 | 0.97× vs PyTorch |

Correctness: optimized path passed all profile unit tests (`small`, `target`, `oddK`) with max_abs ≤ `1e-3`. Baseline provider columns are preserved but pre-skipped because the original dynamic-while Triton baseline aborts the compiler in cannsim for this kernel shape class.
