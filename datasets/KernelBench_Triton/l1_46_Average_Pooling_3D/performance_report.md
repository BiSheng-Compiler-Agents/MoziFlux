# Performance Report

## Cannsim setup

- Baseline source: `46_Average_Pooling_3D.py`
- Optimized source: `opt_46_Average_Pooling_3D.py`
- Simulator tool: `cannsim_local_run(gen_report=True)` with sub-kernel C++ hosts under `cannsim_baseline/` and `cannsim_optimized/`.
- Sub-kernel shape for optimized trace: `N=C=D=H=OD=OH=1, W=128, OW=64, kernel=3, stride=2, padding=1, grid=(1,)`.

## Cannsim trace comparison

| Kernel | Cannsim status | wall cycles | Est. time (cycles × 0.4 ns) | Bottleneck | Notes |
|---|---:|---:|---:|---|---|
| Baseline `BLOCK=256` | compile failed | N/A | N/A | N/A | BiSheng VF stack overflow: total stack object size 9984 > 6144. |
| Baseline `BLOCK=16` diagnostic | record unsafe | N/A | N/A | N/A | Kernel did not reach `[HOST] PASS`; `instr.bin` never became safe, so no reliable `trace_core0.json` was produced. |
| Optimized width-tile | PASS | 2348 | 939.2 ns | MTE3 (1777 cycles) | Trace: `/tmp/cannsim_local/l1_46_avgpool3d_optimized/cannsim_20260625011819_test_kernel/report/trace_core0.json`. |

## Optimized trace table

| Pipeline | Ops | Busy cycles | Lane sum | Window |
|---|---:|---:|---:|---|
| MTE3 | 2 | 1777 | 1777 | [4428,6206] |
| MTE2 | 7 | 1365 | 7952 | [4387,5757] |
| VEC | 1 | 1353 | 1353 | [4403,5756] |
| SCALAR | 100 | 572 | 1847 | [3867,6211] |
| SCALARLDST | 1 | 480 | 480 | [3887,4367] |
| PUSHQ | 5 | 129 | 171 | [4410,5879] |
| RVECEX | 33 | 57 | 224 | [5780,5860] |
| RVECST | 3 | 21 | 27 | [5794,5867] |
| RVECLD | 7 | 20 | 63 | [5779,5830] |
| FLOWCTRL | 2 | 7 | 9 | [6208,6215] |

## Top optimized instructions

| Instruction | Pipe | Count | Total cycles | Avg cycles |
|---|---|---:|---:|---:|
| MOV_SRC_TO_DST_ALIGNv2 | MTE2 | 3 | 4005 | 1335 |
| MOV_SPR_XN | MTE2 | 4 | 3947 | 987 |
| WAIT_FLAG_VEC | MTE3 | 1 | 1452 | 1452 |
| WAIT_FLAG_MTE2 | VEC | 1 | 1353 | 1353 |
| LDP_XI_XJ_XN | SCALAR | 2 | 960 | 480 |
| DC_PRELOAD_XN_IMM | SCALAR | 1 | 494 | 494 |

## Hardware latency

Remote hardware verification ran `profile_kernels.py` on Ascend NPU. Optimized correctness passed on all three shapes, including the persistent-grid dispatch shape.

| label | PyTorch / ACL (ms) | Baseline Triton1 (ms) | Baseline Triton2 (ms) | Optimized Triton (ms) |
|---|---:|---:|---:|---:|
| direct_small | 0.004669 | inf | 0.021454 | 0.011508 |
| nonpow_medium | 0.007235 | inf | 0.179974 | 0.190181 |
| persistent_synthetic | 0.026542 | inf | inf | 8.139362 |

Notes: `Baseline Triton1` is skipped because BiSheng reports VF stack overflow; `Baseline Triton2` is skipped on the persistent synthetic case because its uncapped launch would use `coreDim=280000`. The optimized target-shape legality fix is verified by the persistent synthetic correctness path; the exact full source target is not benchmarked to avoid multi-GB allocation/verification timeout.
