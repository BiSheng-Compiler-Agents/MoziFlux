# Performance Report — l2_54 Conv2d_Multiply_LeakyReLU_GELU

## Cannsim trace setup
- Tool: `cannsim_local_run(gen_report=True)`.
- Baseline job: `/tmp/cannsim_local/l2_54_baseline_small/.../trace_core0.json`.
- Optimized job: `/tmp/cannsim_local/l2_54_opt_small/.../trace_core0.json`.
- Diagnostic sub-kernel: one contiguous HW tile (`HW=128`, fp32), convolution excluded; this isolates the fused post-conv epilogue.
- Cycle conversion: `hardware_time_ns = cycles * 0.4`.

## Trace summary
| Kernel | Events | Wall cycles | Hardware latency (us) | Speedup vs baseline |
|---|---:|---:|---:|---:|
| Baseline Triton1 epilogue | 1668 | 7006 | 2.8024 | 1.00x |
| Optimized NC-loop epilogue | 308 | 3034 | 1.2136 | 2.31x |

## Pipeline busy-cycle comparison
| Pipeline | Baseline busy cyc | Optimized busy cyc | Delta |
|---|---:|---:|---:|
| SCALAR | 8076 | 3709 | -4367 |
| SCALARLDST | 8051 | 3079 | -4972 |
| MTE2 | 749 | 1117 | +368 |
| MTE3 | 648 | 860 | +212 |
| VEC | 0 | 1124 | +1124 |
| RVECEX | 604 | 504 | -100 |
| PUSHQ | 874 | 157 | -717 |
| RVECLD | 38 | 18 | -20 |
| RVECST | 36 | 18 | -18 |
| FLOWCTRL | 9 | 9 | +0 |

## Baseline top instructions
| Instruction | Total cycles |
|---|---:|
| `ST_XD_XN_IMM` | 4024 |
| `LD_XD_XN_IMM` | 2680 |
| `LDP_XI_XJ_XN` | 1442 |
| `LD_XD_XN` | 1347 |
| `STI_XN_IMM` | 1238 |
| `SIGNEXT` | 1060 |
| `ADD_IMM` | 1056 |
| `MOV_SRC_TO_DST_ALIGNv2` | 900 |
| `VF` | 823 |
| `DIV` | 774 |

## Optimized top instructions
| Instruction | Total cycles |
|---|---:|
| `LD_XD_XN_IMM` | 2177 |
| `LDP_XI_XJ_XN` | 1434 |
| `STI_XN_IMM` | 1216 |
| `LD_XD_XN` | 865 |
| `MOV_SRC_TO_DST_ALIGNv2` | 723 |
| `WAIT_FLAG_VEC` | 700 |
| `WAIT_FLAG_MTE2` | 562 |
| `WAIT_FLAG_MTE3` | 562 |
| `MOV_SPR_XN` | 554 |
| `DC_PRELOAD_XN_IMM` | 492 |

## Hardware benchmark (remote_verify)
| label | PyTorch / ACL (ms) | Baseline Triton1 (ms) | Baseline Triton2 (ms) | Optimized Triton (ms) | Opt vs Baseline1 |
|---|---:|---:|---:|---:|---:|
| small_2x64x64 | 0.491543 | 2.548235 | 0.581859 | 0.624289 | 4.08x |
| irregular_3x64x70 | 1.003195 | 4.482508 | 1.063988 | 1.269771 | 3.53x |
| target_64x256 | 89.027657 | 1165.898560 | 79.043129 | 168.715775 | 6.91x |

Correctness: `remote_verify` returned `UNIT_TEST PASS`; max absolute optimized diff was <= `0.000426054` across all benchmark shapes, including the forced persistent dispatch path.

## Interpretation
The cannsim trace shows the optimization reduced wall cycles from 7006 to 3034 (2.31x) and cut total trace events from 1668 to 308. The biggest reductions are from removing per-element `DIV`/`REM` channel-index decomposition and using a single scalar multiplier load per N*C plane.
