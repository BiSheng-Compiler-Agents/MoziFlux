# Performance Report

## Cannsim setup

- Probe: fused Mish + Add + Hardtanh + Scale epilogue only, fp32, `BLOCK_SIZE=4096`, `grid=(1,)`.
- Baseline trace: `/tmp/cannsim_local/l2_16_mish_baseline/cannsim_20260625104315_test_kernel/report/trace_core0.json`.
- Optimized trace: `/tmp/cannsim_local/l2_16_mish_optimized/cannsim_20260625104510_test_kernel/report/trace_core0.json`.
- Note: grid=1 cannsim shows per-tile instruction cost; it cannot show the full-shape benefit of avoiding `coreDim > 65535`.

## Cannsim trace comparison

| Path | wall cycles | est. ns @ 0.4 ns/cyc | events | bottleneck | key critical events |
|---|---:|---:|---:|---|---|
| Baseline direct epilogue | 5906 | 2362.4 | 2562 | MTE3 busy 4111 | WAIT_FLAG_VEC@MTE3 3651, VF@PUSHQ 2659 |
| Optimized persistent epilogue | 6282 | 2512.8 | 2586 | MTE3 busy 4122 | WAIT_FLAG_VEC@MTE3 3668, VF@PUSHQ 2659 |

## Pipeline table

| Path | MTE3 | PUSHQ | RVECEX | SCALAR | SCALARLDST | MTE2 | VEC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline direct | 4111 | 2667 | 2618 | 1810 | 1695 | 1025 | 1019 |
| Optimized persistent | 4122 | 2667 | 2618 | 1805 | 1728 | 1027 | 1023 |

## Dispatch legality

| Shape | Output elements | Baseline tiles @4096 | Optimized route |
|---|---:|---:|---|
| target `(128,64,128,128)` -> `(128,64,256,256)` | 536,870,912 | 131,072 (illegal) | persistent grid capped at 65,535 |

## Hardware latency

Remote verification passed (`UNIT_TEST PASS`). Latencies are milliseconds from `profile_kernels.py`:

| Shape | PyTorch / ACL | Baseline Triton1 | Baseline Triton2 | Optimized Triton | Notes |
|---|---:|---:|---:|---:|---|
| small_direct | 0.025225 | 0.020465 | inf | 0.020470 | direct path |
| medium_direct | 0.077301 | 0.070953 | inf | 0.071068 | direct path |
| target_persistent | 19.330917 | inf | inf | 13.374898 | baseline grid_guard; persistent optimized path |

Target result: optimized Triton is `19.330917 / 13.374898 = 1.445x` faster than PyTorch / ACL on hardware; baseline Triton is not launch-legal at the target because it needs 131,072 programs.
