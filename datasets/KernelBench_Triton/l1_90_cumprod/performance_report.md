# Performance Report: 90_cumprod

## Cannsim setup

- Baseline trace: `/tmp/cannsim_local/l1_90_cumprod_baseline_n64/cannsim_20260625061606_test_kernel/report/trace_core0.json`
- Optimized trace: `/tmp/cannsim_local/l1_90_cumprod_optimized_n64/cannsim_20260625061844_test_kernel/report/trace_core0.json`
- Sub-kernel: one row, `N=64`, `grid=(1,)`, fp32. A larger `N=512` baseline run produced an unsafe-long trace, so the reported comparison uses the largest bounded scalar-loop probe that generated a reliable trace.

## Cannsim trace summary

| Kernel | wall cycles | est. ns (`cycles*0.4`) | x_events | i_events | Bottleneck |
|---|---:|---:|---:|---:|---|
| Baseline scalar loop | 18,934 | 7,573.6 | 1,928 | 390 | MTE3 (10,529 busy cycles) |
| Optimized block scan | 5,037 | 2,014.8 | 765 | 99 | SCALARLDST (3,804 busy cycles) |

## Pipeline comparison

| Pipeline | Baseline busy cycles | Optimized busy cycles | Delta |
|---|---:|---:|---:|
| MTE3 | 10,529 | 940 | -91.1% |
| SCALARLDST | 6,857 | 3,804 | -44.5% |
| PUSHQ | 4,106 | 1,370 | -66.6% |
| SCALAR | 1,865 | 1,472 | -21.1% |
| RVECEX | 896 | 59 | -93.4% |
| RVECLD | 640 | 35 | -94.5% |
| RVECST | 576 | 45 | -92.2% |
| MTE2 | 320 | 700 | +118.8% |

## Top critical instructions

| Kernel | Critical instruction | Count | Total cycles | Notes |
|---|---|---:|---:|---|
| Baseline | `MOV_SRC_TO_DST_ALIGNv2` (MTE3) | 64 | 10,465 | one GM write movement per scalar element dominates |
| Baseline | `ST_XD_XN_IMM` (SCALARLDST) | 192 | 4,288 | scalar loop bookkeeping/spills |
| Optimized | `ST_XD_XN_IMM` (SCALARLDST) | 81 | 7,527 lane-sum | reduced event count but remaining scalar-spill bottleneck |
| Optimized | `VF` (PUSHQ) | 4 | 1,272 | vector prefix-scan dispatch cost |

## Hardware latency (`remote_verify`)

| Shape | PyTorch / ACL (ms) | Baseline Triton1 (ms) | Baseline Triton2 (ms) | Optimized Triton (ms) | Speedup vs Baseline Triton1 |
|---|---:|---:|---:|---:|---:|
| `small_2d` | 0.396398 | 0.119739 | 0.026973 | 0.027556 | 4.35x |
| `medium_2d` | 10.432208 | 2.955858 | 0.314797 | 0.322383 | 9.17x |
| `required_32768x32768` | 13,791.446289 | 2,917.264648 | 299.259674 | 306.718933 | 9.51x |

Unit tests passed for `baseline1`, `baseline2`, and `optimized` on all benchmark shapes, plus the optimized persistent-row dispatch path (`70000x8`).

## Result

Cannsim wall cycles improved from 18,934 to 5,037 on the bounded sub-kernel, a 3.76x per-row probe speedup. Real hardware latency on the required `32768x32768` shape improved from 2,917.264648 ms (editable baseline Triton1) to 306.718933 ms (optimized), a 9.51x speedup; the read-only `baseline2` comparison remains slightly faster at 299.259674 ms.
