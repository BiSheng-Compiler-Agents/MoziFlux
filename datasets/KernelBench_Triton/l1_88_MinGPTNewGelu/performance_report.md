# Performance Report

## Cannsim setup

- Baseline sub-kernel: `_gelu_tanh_kernel`, `BLOCK_SIZE=4096`, `grid=(1,)`, 4096 fp32 elements.
- Optimized sub-kernel: `_gelu_fwd_kernel_even`, `BLOCK_ROWS=4`, `BLOCK_COLS=2048`, `grid=(1,)`, 8192 fp32 elements.
- Trace source: `cannsim_local_run(..., gen_report=True)` with `trace_core0.json` aggregated by `aggregate_trace.py`.
- Cycle conversion: `hardware_time_ns = cycles * 0.4`.

## Trace summary

| Kernel | Elements/program | wall_cycles | cycles/elem | est ns/program | est ns/elem | Bottleneck |
|---|---:|---:|---:|---:|---:|---|
| Baseline | 4,096 | 3,723 | 0.9089 | 1,489.2 | 0.3636 | MTE3 store wait |
| Optimized | 8,192 | 4,951 | 0.6044 | 1,980.4 | 0.2417 | MTE3 store wait |

Normalized throughput improved by **1.50x** (`0.9089 / 0.6044` cycles per element).

## Pipeline table

| Pipeline | Baseline busy_cyc | Optimized busy_cyc | Notes |
|---|---:|---:|---|
| MTE3 | 1,930 | 3,160 | More output elements per program; still bottleneck. |
| SCALAR | 1,791 | 626 | Reduced scalar pipe pressure per larger tile. |
| SCALARLDST | 1,230 | 1,778 | Block-pointer setup cost amortized over 2x elements. |
| MTE2 | 1,018 | 1,071 | Nearly flat despite 2x elements. |
| VEC | 1,012 | 1,062 | Nearly flat despite 2x elements. |
| RVECEX | 431 | 1,447 | More vector math due to 2x elements and tanh lowering. |
| RVECLD | 367 | 1,120 | Scales with vector tile width. |
| RVECST | 375 | 1,014 | Scales with vector tile width. |

## Top critical instructions

| Kernel | Instruction | Pipe | Count | Total cycles | Avg cycles |
|---|---|---|---:|---:|---:|
| Baseline | WAIT_FLAG_VEC | MTE3 | 1 | 1,474 | 1,474 |
| Baseline | RV_VDIV | RVECEX | 64 | 1,088 | 17 |
| Baseline | RV_VEXP | RVECEX | 64 | 1,024 | 16 |
| Optimized | WAIT_FLAG_VEC | MTE3 | 1 | 2,552 | 2,552 |
| Optimized | RV_VDIV | RVECEX | 128 | 2,176 | 17 |
| Optimized | RV_VEXP | RVECEX | 128 | 2,048 | 16 |

## Hardware latency (`remote_verify`)

Correctness: all providers passed all benchmark shapes; optimized persistent dispatch also passed via a forced small-shape unit test (`max_err=1.19209e-07`).

| Shape | PyTorch / ACL ms | Baseline Triton1 ms | Baseline Triton2 ms | Optimized Triton ms | Speedup vs Baseline1 |
|---|---:|---:|---:|---:|---:|
| odd_257x513 | 0.015196 | 0.002650 | 0.004242 | 0.005760 | 0.46x |
| medium_1024x2048 | 0.064670 | 0.020352 | 0.019407 | 0.020648 | 0.99x |
| target_8192x8192 | 4.100237 | 0.537209 | 0.446032 | 0.476982 | 1.13x |

Target-shape hardware latency improved from **0.537209 ms → 0.476982 ms** versus the editable baseline (`1.13x`).
