# Performance Report

## Cannsim setup

- Baseline probe: `_mse_partial_sum_kernel`, grid=(1,), BLOCK_SIZE=4096, fp32 inputs, one 4096-element tile.
- Optimized Triton fallback probe: `_mse_stage1_kernel`, grid=(1,), BLOCK_SIZE=4096, fp32 inputs, one 4096-element tile.
- Tool: `cannsim_local_run(..., gen_report=True, soc_version="Ascend950")`.
- Trace files:
  - Baseline: `/tmp/cannsim_local/mse94_baseline/cannsim_20260625072921_test_kernel/report/trace_core0.json`
  - Optimized fallback: `/tmp/cannsim_local/mse94_optimized/cannsim_20260625073113_test_kernel/report/trace_core0.json`

## Cannsim trace comparison

| Metric | Baseline | Optimized fallback | Change |
|---|---:|---:|---:|
| wall_cycles | 5020 | 3886 | -22.6% |
| x_events | 1135 | 545 | -52.0% |
| i_events | 48 | 20 | -58.3% |
| hardware time (cycles × 0.4 ns) | 2008.0 ns | 1554.4 ns | -453.6 ns |

## Pipeline utilization

| Pipeline | Baseline busy_cyc | Optimized fallback busy_cyc | Change |
|---|---:|---:|---:|
| SCALARLDST | 1834 | 1791 | -2.3% |
| PUSHQ | 1788 | 664 | -62.9% |
| MTE2 | 1601 | 1070 | -33.2% |
| VEC | 1544 | 1042 | -32.5% |
| SCALAR | 1463 | 1447 | -1.1% |
| RVECEX | 573 | 277 | -51.7% |
| RVECLD | 460 | 226 | -50.9% |
| MTE3 | 287 | 336 | +17.1% |
| RVECST | 180 | 18 | -90.0% |
| FLOWCTRL | 7 | 7 | 0.0% |

## Top bottlenecks

| Kernel | Bottleneck | Critical instruction |
|---|---|---|
| Baseline | SCALARLDST 1834 busy cycles | `RV_VCADD` 2816 total cycles; `VF`/`WAIT_FLAG_MTE2` also high |
| Optimized fallback | SCALARLDST 1791 busy cycles | `MOV_SRC_TO_DST_ALIGNv2` 2076 total cycles; fewer vector/reduction instructions |

## Hardware latency (`remote_verify`)

| label | PyTorch / ACL (ms) | Baseline Triton1 (ms) | Baseline Triton2 (ms) | Optimized Triton (ms) |
|---|---:|---:|---:|---:|
| small_single | 0.009826 | 0.037230 | 0.037604 | 0.012038 |
| direct_1m | 0.007455 | 0.034378 | 0.035619 | 0.011723 |
| persistent_1g | 5.277135 | inf (grid_guard) | 5.439062 | 5.284158 |

Correctness: all optimized test labels passed. The source baseline is skipped at `persistent_1g` because its launch grid is 262,144 blocks, above Ascend's 65,535 FFTS cap.
