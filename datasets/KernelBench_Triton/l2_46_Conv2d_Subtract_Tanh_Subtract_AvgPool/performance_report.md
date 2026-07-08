# Performance Report

## Cannsim setup

- Baseline trace: `/tmp/cannsim_local/l2_46_baseline_epilogue_tiny/cannsim_20260629213021_test_kernel/report/trace_core0.json`
- Optimized fallback trace: `/tmp/cannsim_local/l2_46_opt_epilogue_tiny/cannsim_20260629213216_test_kernel/report/trace_core0.json`
- Sub-kernel: one AI-vector program, `BLOCK_HW=16`, `K=2`, fp32 input, correctness checked in C++ host.
- Cycle conversion: `hardware_time_ns = cycles * 0.4`.

## Cannsim trace summary

| Kernel | Events | Wall cycles (max ts) | Hardware latency (ns) | Total event dur | Bottleneck |
|---|---:|---:|---:|---:|---|
| Baseline fused Triton epilogue | 845 | 10,837 | 4,334.8 | 22,767 | SCALARLDST 56.20% |
| Optimized Triton fallback candidate | 1,063 | 11,058 | 4,423.2 | 26,844 | SCALARLDST 53.19% |

## Pipeline breakdown

| Pipeline | Baseline dur | Baseline % | Optimized fallback dur | Optimized fallback % |
|---|---:|---:|---:|---:|
| SCALARLDST | 12,795 | 56.20% | 14,278 | 53.19% |
| SCALAR | 4,889 | 21.47% | 6,370 | 23.73% |
| PUSHQ | 3,034 | 13.33% | 3,701 | 13.79% |
| RVECEX | 654 | 2.87% | 1,057 | 3.94% |
| MTE3 | 961 | 4.22% | 961 | 3.58% |
| RVECLD | 248 | 1.09% | 263 | 0.98% |
| RVECST | 153 | 0.67% | 180 | 0.67% |
| MTE2 | 20 | 0.09% | 20 | 0.07% |
| FLOWCTRL | 9 | 0.04% | 9 | 0.03% |
| RVECSU | 4 | 0.02% | 5 | 0.02% |

## Cannsim conclusion

The custom Triton epilogue is scalar/indexing bound; the attempted Triton fallback does not improve the micro-trace (`10,837 -> 11,058` wall cycles). Therefore the production optimized `ModelNew` dispatches the standard activation/pool epilogue through CANN/ACL and retains the grid-cap-safe Triton fallback only as a tested fallback path.

## Remote hardware benchmark

Remote verification directory: `/home/s00929845/kernel_verify/l2_46_Conv2d_Subtract_Tanh_Subtract_AvgPool_1782769131`.

| Label | PyTorch / ACL (ms) | Baseline Triton1 (ms) | Baseline Triton2 (ms) | Optimized Triton (ms) | Baseline1 / Optimized |
|---|---:|---:|---:|---:|---:|
| small_direct | 0.419981 | 10.864983 | 10.875407 | 0.420399 | 25.84x |
| irregular_direct | 0.364125 | 6.727750 | 6.755273 | 0.362723 | 18.55x |
| default_persistent_boundary | 85.747398 | 0.029898 | 0.028505 | 0.028576 | 1.05x |

## Correctness

`remote_verify` reported `test_passed=true` and `bench_passed=true`.

```
UNIT small_direct optimized: max_abs=0 PASS
UNIT irregular_direct optimized: max_abs=0 PASS
UNIT default_persistent_boundary optimized: max_abs=0 PASS
UNIT forced_direct opt_triton_direct: max_abs=0 PASS
UNIT forced_persistent opt_triton_persistent: max_abs=0 PASS
UNIT_TEST PASS
```
