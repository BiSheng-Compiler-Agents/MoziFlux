# Performance Report

## Cannsim setup

- Tool: `cannsim_local_run(gen_report=True)`.
- Probe shape: sub-kernel, one program, `N=1`, `C=32`, `H=8`, `W=64`.
- Baseline probe: `_fused_min_sum_gelu_add_bias` copied from the editable input kernel.
- Optimized probe: `_add_bias_expand_kernel` epilogue from `opt_36_ConvTranspose2d_Min_Sum_GELU_Add.py`.
- Cycle-to-time conversion: `cycles * 0.4 ns`.

## Cannsim trace summary

| Path | Cannsim result | trace_core0 | wall cycles | HW latency | Bottleneck |
|---|---|---:|---:|---:|---|
| Baseline Triton post-kernel | Build failed in BiSheng (`Not all operands are collapsed`, `Collapser.cpp` assertion) | unavailable | n/a | n/a | nested `tl.min` reduction compile abort |
| Optimized Triton add epilogue | PASS | `/tmp/cannsim_local/l2_36_opt_add/cannsim_20260629183139_test_kernel/report/trace_core0.json` | 3,119 | 1.2476 us | SCALAR / SCALARLDST |

Optimized epilogue pipeline table from `trace_summary.txt`:

| Pipeline | Ops | Busy cycles | Window |
|---|---:|---:|---|
| SCALAR | 197 | 1,919 | [3913,7028] |
| SCALARLDST | 4 | 1,717 | [3933,5732] |
| MTE3 | 2 | 1,263 | [5759,7023] |
| MTE2 | 5 | 872 | [5717,6590] |
| VEC | 1 | 839 | [5750,6589] |
| PUSHQ | 2 | 74 | [5754,6661] |
| RVECEX | 17 | 28 | [6613,6642] |
| RVECLD | 17 | 26 | [6612,6638] |
| RVECST | 16 | 24 | [6625,6649] |
| FLOWCTRL | 2 | 7 | [7025,7032] |

Top optimized epilogue critical instructions:

| Instruction | Pipe | Count | Total cycles | Avg cycles |
|---|---|---:|---:|---:|
| LD_XD_XN_IMM | SCALARLDST | 4 | 2,193 | 548 |
| LDP_XI_XJ_XN | SCALAR | 4 | 1,953 | 488 |
| MOV_SRC_TO_DST_ALIGNv2 | MTE2 | 2 | 1,523 | 762 |
| MOV_SPR_XN | MTE2 | 3 | 1,492 | 497 |
| STI_XN_IMM | SCALAR | 2 | 1,267 | 634 |
| WAIT_FLAG_VEC | MTE3 | 1 | 903 | 903 |
| WAIT_FLAG_MTE2 | VEC | 1 | 839 | 839 |

## Remote hardware latency

`remote_verify(run_test=True, run_bench=True)` passed optimized correctness on all benchmark shapes. Baseline Triton providers were kept visible but pre-skipped as `inf` because the cannsim probe showed the nested reduction can abort the Ascend compiler and poison the process.

| Label | PyTorch / ACL (ms) | Baseline Triton1 | Baseline Triton2 | Optimized Triton (ms) | torch/opt ratio |
|---|---:|---:|---:|---:|---:|
| small_default | 0.359492 | inf | inf | 0.359370 | 1.000x |
| small_multibias | 0.381217 | inf | inf | 0.363572 | 1.049x |
| target | 31.850307 | inf | inf | 32.052872 | 0.994x |

Result: optimized correctness is verified. The optimized Triton epilogue is faster on the multi-bias small shape and slightly slower than pure ACL on the target shape, so the primary value of this optimization is correctness/safety and avoiding the baseline compiler-aborting custom reduction.
