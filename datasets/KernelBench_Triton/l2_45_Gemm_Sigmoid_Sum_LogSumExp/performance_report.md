# Performance Report

## Correctness and hardware benchmark

Remote hardware verification passed (`UNIT_TEST PASS`) for all providers and all benchmark shapes. Latencies below are milliseconds from `profile_kernels.py` on the remote Ascend NPU.

| label | PyTorch / ACL | Baseline Triton1 | Baseline Triton2 | Optimized Triton | Speedup vs Baseline Triton1 |
|---|---:|---:|---:|---:|---:|
| default_B128_K10_H20 | 0.300901 | 0.353859 | 0.514104 | 0.078209 | 4.52x |
| boundary_B17_K10_H20 | 0.283041 | 0.097949 | 0.266649 | 0.073361 | 1.34x |
| aligned_B64_K16_H32 | 0.285604 | 0.152980 | 0.229801 | 0.065873 | 2.32x |

Default-shape hardware latency improved from **0.353859 ms** to **0.078209 ms**.

## Cannsim trace comparison

Cannsim was run with sub-kernel hosts and `gen_report=True`:

- Baseline fused scalar/vector sub-kernel: `/tmp/cannsim_local/l2_45_baseline_fused/cannsim_20260629210530_test_kernel/report/trace_core0.json`
- Optimized Cube row-sum sub-kernel: `/tmp/cannsim_local/l2_45_opt_dot_rowsum/cannsim_20260629210812_test_kernel/report/trace_core0.json`

### Summary

| Metric | Baseline fused | Optimized row-sum | Delta |
|---|---:|---:|---:|
| wall_cycles | 13,315 | 6,132 | -53.9% |
| trace events | 12,277 | 1,429 | -88.4% |
| dominant bottleneck | PUSHQ | SCALARLDST | shifted |
| hardware time estimate (`cycles * 0.4 ns`) | 5,326 ns | 2,453 ns | -53.9% |

### Pipeline utilization

| Pipeline | Baseline busy cycles | Optimized busy cycles | Observation |
|---|---:|---:|---|
| PUSHQ | 9,163 | 1,961 | Major dispatch/queue pressure removed by Cube GEMM tile. |
| RVECEX | 5,081 | 270 | Vector multiply/add reduction mostly eliminated. |
| RVECST | 4,766 | 346 | Far less vector local-buffer store traffic. |
| RVECLD | 4,747 | 322 | Far less vector local-buffer load traffic. |
| SCALARLDST | 3,624 | 3,160 | Still the optimized bottleneck, but lower absolute cost. |
| MTE2 | 1,002 | 1,775 | More visible after removing vector work; includes dot-tile movement. |
| CUBE | 0 | 158 | Optimized kernel activates Cube via `tl.dot`. |
| MTE3 | 307 | 2,560 | Row-sum store and pipeline waits become visible after vector reduction is removed. |

### Top instruction changes

| Instruction | Baseline total cycles | Optimized total cycles | Interpretation |
|---|---:|---:|---|
| ST_XD_XN_IMM | 155,060 | 71,812 | Scalar local stores reduced substantially. |
| RV_VCADD | 23,254 | absent from top list | Vector add tree removed from GEMM path. |
| RV_VMUL | 8,208 | absent from top list | Elementwise multiply GEMM emulation removed. |
| VF | 8,763 | 1,700 | Queue pressure reduced. |
| RV_VLDI | 20,281 | 1,693 | Vector local loads reduced. |

## Interpretation

The baseline spends most time in vector and queue activity because it implements GEMM as broadcast multiply plus `tl.sum`. The optimized kernel uses `tl.dot` for the `[16,16] x [16,32]` tile, reducing sub-kernel wall cycles by **53.9%** and lowering default hardware latency by **4.52x** versus Baseline Triton1.
