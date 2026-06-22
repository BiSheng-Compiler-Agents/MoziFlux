# Performance Report — 36_RMSNorm_

## Correctness and hardware latency

Remote Ascend verification passed for the optimized kernel on all profiler shapes.

| label | PyTorch / ACL (ms) | Baseline Triton1 (ms) | Baseline Triton2 (ms) | Optimized Triton (ms) |
|---|---:|---:|---:|---:|
| small | 0.012479 | 0.018679 | 0.003883 | 0.003948 |
| irregular | 0.013495 | 0.027542 | 0.005017 | 0.004988 |
| medium | 0.112367 | inf (coreDim guard) | 0.077636 | 0.167221 |
| target | 30.311148 | inf (coreDim guard) | inf (coreDim guard) | 14.053106 |

Target hardware latency: **14.053106 ms** optimized vs **30.311148 ms** PyTorch/ACL (**2.157x faster**). Existing Triton baselines exceed Ascend `coreDim <= 65535` at target size and are guarded in `profile_kernels.py` to avoid poisoning the process.

## Cannsim trace comparison

Sub-kernel runs use `grid=(1,)`. Baseline sub-kernel processes 64 output elements; optimized sub-kernel processes 64 channels × 128 HW positions = 8192 output elements.

| kernel | trace path | wall cycles | output elems | cycles/elem | dominant pipeline | top critical instruction |
|---|---|---:|---:|---:|---|---|
| baseline | `/tmp/cannsim_local/rmsnorm36_baseline/cannsim_20260624210819_test_kernel/report/trace_core0.json` | 4131 | 64 | 64.546875 | SCALARLDST 2020 busy cycles | `ST_XD_XN_IMM` 3158 total cycles |
| optimized | `/tmp/cannsim_local/rmsnorm36_optimized_persistent/cannsim_20260624211330_test_kernel/report/trace_core0.json` | 6650 | 8192 | 0.811768 | MTE3 3364 busy cycles | `WAIT_FLAG_VEC` 2758 total cycles |

Normalized cannsim throughput improved from **64.55 cycles/elem** to **0.812 cycles/elem** (**79.51x**). Raw optimized wall cycles are higher because a single optimized program intentionally performs 128× more row work than the baseline sub-kernel.

## Pipeline tables

### Baseline

| pipeline | ops | busy cycles |
|---|---:|---:|
| SCALARLDST | 21 | 2020 |
| PUSHQ | 10 | 1596 |
| MTE2 | 9 | 878 |
| MTE3 | 2 | 874 |
| SCALAR | 254 | 758 |
| VEC | 3 | 757 |
| RVECEX | 16 | 120 |
| RVECLD | 10 | 50 |
| RVECST | 5 | 45 |

### Optimized

| pipeline | ops | busy cycles |
|---|---:|---:|
| MTE3 | 2 | 3364 |
| PUSHQ | 10 | 3087 |
| RVECLD | 516 | 2485 |
| SCALARLDST | 90 | 2143 |
| RVECEX | 904 | 1597 |
| RVECST | 260 | 1393 |
| MTE2 | 9 | 1345 |
| VEC | 3 | 1336 |
| SCALAR | 392 | 982 |

## Verification commands/results

- `cannsim_local_run(local_dir=.../cannsim_baseline, job_name="rmsnorm36_baseline")`: success, `[HOST] PASS`.
- `cannsim_local_run(local_dir=.../cannsim_optimized, job_name="rmsnorm36_optimized_persistent")`: success, `[HOST] PASS`.
- `remote_verify(..., run_test=True, run_bench=True)`: `test_passed=true`, `bench_passed=true`.
