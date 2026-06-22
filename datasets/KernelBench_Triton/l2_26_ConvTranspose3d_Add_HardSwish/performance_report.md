# Performance Report

## Cannsim setup

Sub-kernel simulation used `cannsim_local_run(gen_report=True)` with one launched program on Ascend950. Baseline compiles the original direct epilogue kernel with `BLOCK=2048`; optimized compiles the persistent epilogue with `BLOCK=4096` and `n_programs=1` for a one-tile trace.

Trace files:
- Baseline: `/tmp/cannsim_local/l2_26_hswish_baseline/cannsim_20260625124139_test_kernel/report/trace_core0.json`
- Optimized: `/tmp/cannsim_local/l2_26_hswish_opt/cannsim_20260625124323_test_kernel/report/trace_core0.json`

## Cannsim trace summary

| Path | Probe | Elements/tile | wall_cycles | sim latency (ns) | Bottleneck | Bottleneck busy cycles | Notes |
|---|---:|---:|---:|---:|---|---:|---|
| Baseline Triton1 direct epilogue | grid=1 | 2,048 | 3,701 | 1,480.4 | MTE3 | 1,892 | Direct target launch would need 131,072 programs and exceed grid cap. |
| Optimized persistent epilogue | grid=1 | 4,096 | 4,560 | 1,824.0 | MTE3 | 2,395 | Per-tile trace includes persistent loop overhead; full-shape benefit is legal capped dispatch. |

Normalized throughput:

| Path | cycles / element | ns / element |
|---|---:|---:|
| Baseline direct | 1.807 | 0.723 |
| Optimized persistent | 1.113 | 0.445 |

## Pipeline tables

### Baseline direct epilogue

| Pipeline | ops | busy_cyc | Notes |
|---|---:|---:|---|
| MTE3 | 2 | 1,892 | Bottleneck, store to GM |
| SCALAR | 110 | 1,813 | Launch/index/control overhead |
| SCALARLDST | 3 | 1,718 | Scalar load/store setup |
| MTE2 | 5 | 1,015 | Loads from GM |
| VEC | 1 | 1,002 | Wait on MTE2 |
| RVECEX | 582 | 460 | HardSwish vector math |

Top critical instructions: `MOV_SRC_TO_DST_ALIGNv2` 1,998 cycles, `WAIT_FLAG_VEC@MTE3` 1,488 cycles, `WAIT_FLAG_MTE2` 1,002 cycles.

### Optimized persistent epilogue

| Pipeline | ops | busy_cyc | Notes |
|---|---:|---:|---|
| MTE3 | 2 | 2,395 | Bottleneck, store to GM |
| SCALAR | 132 | 1,826 | Includes persistent tile loop control |
| SCALARLDST | 4 | 1,705 | Scalar setup |
| MTE2 | 7 | 1,077 | Loads from GM |
| VEC | 2 | 1,065 | Waits on MTE2/MTE3 |
| RVECEX | 1,157 | 843 | HardSwish vector math over 2x tile size |

Top critical instructions: `LD_XD_XN_IMM` 2,182 cycles, `MOV_SRC_TO_DST_ALIGNv2` 2,108 cycles, `WAIT_FLAG_VEC@MTE3` 1,944 cycles.

## Hardware benchmark

`remote_verify(run_test=True, run_bench=True)` passed. Correctness covered the optimized direct path (`small_direct`, `medium_direct`) and persistent path (`target_persistent`):

```text
TEST optimized small_direct PASS max_abs=0
TEST optimized medium_direct PASS max_abs=0
TEST optimized target_persistent PASS max_abs=0
UNIT_TEST PASS
```

| label | PyTorch / ACL (ms) | Baseline Triton1 (ms) | Baseline Triton2 (ms) | Optimized Triton (ms) | Optimized vs PyTorch / ACL |
|---|---:|---:|---:|---:|---:|
| small_direct | 0.023062 | 0.014778 | inf | 0.015268 | 1.511x |
| medium_direct | 0.099436 | 0.086471 | inf | 0.078851 | 1.261x |
| target_persistent | 13.696477 | inf | inf | 7.543422 | 1.816x |

Baseline Triton1 is `grid_guard` at the target shape because its direct fp32 epilogue would launch 131,072 programs. Baseline Triton2 is read-only and kept parser-visible as `inf`/skip.
