# Performance Report

## Cannsim setup

- Baseline sub-kernel: `_relu_hswish_inplace_kernel`, `BLOCK_SIZE=8192`, grid `(1,)`, fp32 in-place activation tile.
- Optimized sub-kernel: `_relu_hswish_direct_kernel`, `BLOCK_SIZE=8192`, grid `(1,)`, same fp32 in-place activation tile.
- Trace files:
  - Baseline: `/tmp/cannsim_local/l2_57_baseline/cannsim_20260629235912_test_kernel/report/trace_core0.json`
  - Optimized: `/tmp/cannsim_local/l2_57_optimized_v2/cannsim_20260630000337_test_kernel/report/trace_core0.json`
- Hardware time conversion: `cycles * 0.4 ns`.

## Cannsim summary

| Metric | Baseline | Optimized | Delta |
|---|---:|---:|---:|
| wall_cycles | 4,972 | 4,405 | -11.40% |
| hardware_time_ns | 1,988.8 | 1,762.0 | -226.8 ns |
| x_events | 2,549 | 1,906 | -25.23% |
| i_events | 8 | 8 | 0 |
| speedup | 1.00x | 1.129x | +12.87% |

## Pipeline table

| Pipeline | Baseline busy_cyc | Optimized busy_cyc | Delta |
|---|---:|---:|---:|
| MTE3 | 3,175 | 2,623 | -17.39% |
| SCALAR | 1,794 | 1,779 | -0.84% |
| SCALARLDST | 1,711 | 1,697 | -0.82% |
| PUSHQ | 1,509 | 976 | -35.32% |
| RVECEX | 1,461 | 929 | -36.41% |
| RVECLD | 1,132 | 883 | -22.00% |
| MTE2 | 1,067 | 1,069 | +0.19% |
| VEC | 1,061 | 1,063 | +0.19% |
| RVECST | 1,047 | 780 | -25.50% |
| FLOWCTRL | 7 | 7 | 0 |

## Top instruction changes

| Instruction | Baseline total_cyc | Optimized total_cyc | Notes |
|---|---:|---:|---|
| WAIT_FLAG_VEC | 2,547 | 2,018 | Lower vector tail wait after fewer vector ops |
| RV_VADDS | 2,688 | 1,792 | Fewer vector add lanes from algebraic simplification |
| RV_VMINS | 2,304 | 1,536 | Fewer min operations |
| RV_VMAXS | 2,304 | 0 | Removed separate ReLU max |
| RV_VSEL | 1,536 | 1,536 | Retained selection for piecewise zeroing |
| VF / PUSHQ | 1,502 | not top-12 | PUSHQ pressure reduced |

## Remote hardware verification

`remote_verify` completed with `test_passed=True` and `bench_passed=True`.

| label | PyTorch / ACL (ms) | Baseline Triton1 (ms) | Baseline Triton2 | Optimized Triton (ms) | Opt vs Baseline1 |
|---|---:|---:|---:|---:|---:|
| default | 46.918804 | 18.718069 | inf (reference read forbidden) | 17.887016 | 1.046x |
| small_irregular | 0.535934 | 0.269839 | inf (reference read forbidden) | 0.245997 | 1.097x |
| batch1 | 0.206875 | 0.115719 | inf (reference read forbidden) | 0.123684 | 0.936x |

Correctness results: optimized direct path passed all benchmark shapes with `max_abs=0`; optimized forced-persistent path passed with `max_abs=0`.
