# Performance Report

## Verification summary

- `cannsim_local_run` completed for baseline and optimized sub-kernel microprobes.
- `remote_verify` completed on Ascend hardware: `test_passed=true`, `bench_passed=true`.
- Unit test max absolute error: <= `1.78814e-07` on benchmark shapes; all forced dispatch-path tests passed.

## Cannsim trace comparison

Microprobe scope: fused pointwise epilogue only, `BLOCK_SIZE=256`, `grid=(1,)`, fp32 input/output. This intentionally scales the trace down so cannsim finishes; it preserves the scalar-indexing bottleneck and instruction mix of the epilogue.

| Metric | Baseline | Optimized | Change |
|---|---:|---:|---:|
| wall cycles | 16900 | 2171 | 7.78x faster / -87.2% |
| hardware time @ 0.4 ns/cycle | 6760.0 ns | 868.4 ns | -5891.6 ns |
| x_events | 5907 | 153 | -97.4% |
| i_events | 536 | 7 | -98.7% |
| dominant pipeline | SCALARLDST 14902 busy cyc | SCALARLDST 1314 busy cyc | -91.2% |
| SCALAR busy cycles | 12126 | 567 | -95.3% |
| MTE2 busy cycles | 963 | 954 | unchanged memory floor |
| MTE3 busy cycles | 974 | 777 | -20.2% |
| RVECEX busy cycles | 119 | 110 | -7.6% |

### Baseline cannsim pipeline table

| Pipeline | Ops | Busy cycles | Notes |
|---|---:|---:|---|
| SCALARLDST | 1027 | 14902 | bottleneck; 512 `ST_XD_XN_IMM` |
| SCALAR | 4794 | 12126 | 513 `DIV` + 513 `REM`, sign-extend/add overhead |
| PUSHQ | 7 | 1106 | queue overhead |
| MTE3 | 2 | 974 | store |
| MTE2 | 3 | 963 | load |
| RVECEX | 49 | 119 | activation math |

Top baseline instruction costs: `ST_XD_XN_IMM` 16112 total cycles, `SIGNEXT` 8224, `ADD_IMM` 4136, `DIV` 3078, `REM` 3078.

### Optimized cannsim pipeline table

| Pipeline | Ops | Busy cycles | Notes |
|---|---:|---:|---|
| SCALARLDST | 3 | 1314 | bottleneck, now startup/load dominated |
| MTE2 | 3 | 954 | contiguous GM->UB load floor |
| VEC | 1 | 947 | waits on MTE2 |
| MTE3 | 2 | 777 | store |
| SCALAR | 83 | 567 | per-element div/rem eliminated |
| PUSHQ | 2 | 380 | lower dispatch pressure |
| RVECEX | 47 | 110 | activation math retained |

Top optimized instruction costs: `LD_XD_XN` 1527 total cycles, `LDP_XI_XJ_XN` 1445, `MOV_SRC_TO_DST_ALIGNv2@MTE2` 951, `WAIT_FLAG_MTE2` 947.

Trace paths:
- Baseline: `/tmp/cannsim_local/l2_48_baseline_small/cannsim_20260629215855_test_kernel/report/trace_core0.json`
- Optimized: `/tmp/cannsim_local/l2_48_optimized_small/cannsim_20260629220129_test_kernel/report/trace_core0.json`

## Hardware latency (`remote_verify`)

| label | PyTorch / ACL ms | Baseline Triton ms | Optimized Triton ms | Speedup vs baseline | Optimized vs PyTorch / ACL |
|---|---:|---:|---:|---:|---:|
| small | 0.315372 | 2.079632 | 0.212052 | 9.81x | 1.49x |
| medium | 2.155447 | 22.577723 | 1.982612 | 11.39x | 1.09x |
| default | 46.876396 | 973.403809 | 35.403500 | 27.49x | 1.32x |

Default-shape result: optimized Triton is `27.49x` faster than the editable baseline and `1.32x` faster than the PyTorch/ACL reference chain.
