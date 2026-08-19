# Performance Report

## Correctness

Remote hardware verification passed for all providers and benchmark shapes. Optimized max absolute error vs PyTorch/ACL:

| Shape | Max abs diff |
|---|---:|
| tiny | 1.19209e-07 |
| irregular | 8.9407e-08 |
| default | 1.78814e-07 |
| forced persistent unit | 1.19209e-07 |

## Hardware latency (`remote_verify`, ms)

| label | PyTorch / ACL | Baseline Triton1 | Baseline Triton2 | Optimized Triton | Speedup vs Baseline Triton1 | Torch/Opt ratio |
|---|---:|---:|---:|---:|---:|---:|
| tiny | 0.146255 | 0.144988 | 0.136237 | 0.138770 | 1.0448x | 1.0539x |
| irregular | 0.159240 | 0.160164 | 0.151961 | 0.154202 | 1.0387x | 1.0327x |
| default | 55.902035 | 60.298351 | 56.362362 | 57.106045 | 1.0559x | 0.9789x |

The optimized Triton epilogue improves the editable baseline path by 3.7-5.3%. For the default full model, ACL/PyTorch remains slightly faster than the optimized Triton epilogue path because the convolution dominates total latency.

## Cannsim sub-kernel trace comparison

Trace target: post-conv `mish -> tanh` epilogue only, `BLOCK_SIZE=4096`, `grid=(1,1,1)`, fp32 buffer. Cycle-to-time conversion uses `0.4 ns/cycle`.

| Metric | Baseline | Optimized | Delta |
|---|---:|---:|---:|
| wall_cycles | 5,293 | 4,619 | -12.73% |
| hardware time | 2,117.2 ns | 1,847.6 ns | -269.6 ns |
| x_events | 2,166 | 1,644 | -24.1% |
| trace JSON bytes | 622,919 | 474,369 | -23.8% |

### Pipeline utilization

| Pipeline | Baseline busy_cyc | Optimized busy_cyc | Delta | Note |
|---|---:|---:|---:|---|
| MTE3 | 3,507 | 2,839 | -19.0% | bottleneck remains vector-store wait |
| PUSHQ | 2,059 | 1,385 | -32.7% | fewer vector instruction groups |
| RVECEX | 2,013 | 1,340 | -33.4% | fewer exp/mul/add operations |
| SCALAR | 1,784 | 1,778 | -0.3% | unchanged launch/index overhead |
| SCALARLDST | 1,216 | 1,218 | +0.2% | unchanged |
| MTE2 | 1,018 | 1,017 | -0.1% | unchanged one input tile load |
| RVECST | 575 | 537 | -6.6% | reduced vector temporaries |
| RVECLD | 570 | 567 | -0.5% | unchanged |

### Top instruction deltas

| Instruction | Baseline total_cyc | Optimized total_cyc | Delta | Rationale |
|---|---:|---:|---:|---|
| RV_VEXP | 3,072 | 2,048 | -33.3% | one exponential removed from Mish path |
| RV_VADDS | 2,688 | 1,792 | -33.3% | simplified activation algebra |
| RV_VMULS | 2,560 | 2,048 | -20.0% | fewer scalar-vector multiply steps |
| RV_VMUL | 2,048 | 1,024 | -50.0% | final tanh manual sequence removed |
| RV_VDIV | 2,176 | 3,264 | +50.0% | stable ratio formulation uses extra divisions but is still net faster |
| WAIT_FLAG_VEC | 3,052 | 2,380 | -22.0% | lower vector work reduces MTE3 wait |
| VF/PUSHQ | 2,054 | 1,381 | -32.8% | smaller vector instruction stream |

## Trace artifacts

- Baseline trace: `/tmp/cannsim_local/l2_47_mish_tanh_baseline/cannsim_20260629214419_test_kernel/report/trace_core0.json`
- Optimized trace: `/tmp/cannsim_local/l2_47_mish_tanh_opt/cannsim_20260629214614_test_kernel/report/trace_core0.json`
- Local summaries: `cannsim_baseline_trace_summary.txt`, `cannsim_opt_trace_summary.txt`
