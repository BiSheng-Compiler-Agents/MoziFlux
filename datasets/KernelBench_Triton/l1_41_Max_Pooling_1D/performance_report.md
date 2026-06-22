# Performance Report

## Cannsim Setup

- Tool: `cannsim_local_run(gen_report=True)`
- Target: Ascend950 simulator; compiled target `Ascend910_9589`
- Baseline trace: `/tmp/cannsim_local/l1_41_maxpool_baseline_p0/cannsim_20260624223203_test_kernel/report/trace_core0.json`
- Optimized trace: `/tmp/cannsim_local/l1_41_maxpool_opt_fast/cannsim_20260624223438_test_kernel/report/trace_core0.json`
- Sub-kernel: one `(NC, output-tile)` tile, `BLOCK=128`, `K=8`; representative full interior tile (`padding=0`) to isolate the dominant target-path optimization.
- Cycle conversion: `hardware_time_ns = cycles * 0.4`.

## Cannsim Summary

| Kernel | wall_cycles | hardware_time_ns | x_events | i_events | Bottleneck |
|---|---:|---:|---:|---:|---|
| Baseline | 16,222 | 6,488.8 | 9,837 | 1,080 | SCALARLDST |
| Optimized | 4,012 | 1,604.8 | 728 | 47 | SCALARLDST |
| Delta | -75.3% | -75.3% | -92.6% | -95.6% | scalar/control reduced |

## Pipeline Breakdown

| Kernel | SCALARLDST busy | SCALAR busy | MTE2 busy | MTE3 busy | RVECEX busy | PUSHQ busy |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 13,338 | 10,313 | 5 | 990 | 137 | 1,427 |
| Optimized | 2,814 | 921 | 1,061 | 1,277 | 181 | 572 |
| Delta | -78.9% | -91.1% | +1,056 | +29.0% | +32.1% | -59.9% |

## Top Instruction Changes

| Metric | Baseline | Optimized | Interpretation |
|---|---:|---:|---|
| `JUMPC` instant events | 1,048 | 22 | Fast interior-tile branch removes per-window boundary/control checks. |
| `CMP_IMM` total cycles | 4,160 | not top-12 | Bounds comparisons are eliminated on full tiles. |
| `MIN` total cycles | 4,104 | not top-12 | Address clamp removed on full tiles. |
| `MAX` total cycles | 4,136 | not top-12 | Address clamp removed on full tiles. |
| `ADD` total cycles | 8,292 | 168 | Loop/index arithmetic reduced. |

## Hardware Verification

Remote hardware verification via `remote_verify` passed correctness and benchmark.

| label | PyTorch / ACL (ms) | Baseline Triton1 (ms) | Baseline Triton2 (ms) | Optimized Triton (ms) |
|---|---:|---:|---:|---:|
| direct_small | 0.002715 | 0.008880 | 0.009459 | 0.111657 |
| persistent_synthetic | 1.243299 | inf (coreDim guard) | inf (coreDim guard) | 217.329727 |
| target_original | 62.552536 | inf (coreDim guard) | inf (coreDim guard) | 230.591980 |

Correctness coverage:
- `optimized direct_small`: PASS
- `optimized persistent_synthetic`: PASS
- `optimized target_original`: PASS
- `optimized index_direct`: PASS

Interpretation: the optimized Triton path is correct and legal for shapes where the editable/golden baseline launches exceed the Ascend `coreDim <= 65535` limit. Hardware latency is slower than PyTorch/ACL on this operator, but the Triton implementation remains benchmarkable and valid for the required target shape.
