# Performance Report: 66_Matmul_Dropout_Mean_Softmax

## Correctness and hardware benchmark

Remote Ascend verification passed.  `Baseline Triton2` is kept visible but marked unavailable because the sandbox explicitly forbids reading `base_*.py`.

| label | PyTorch / ACL (ms) | Baseline Triton1 (ms) | Baseline Triton2 | Optimized Triton (ms) | Speedup vs Baseline1 |
|---|---:|---:|---:|---:|---:|
| tiny | 0.015772 | 0.011796 | inf | 0.009842 | 1.20x |
| default | 0.015760 | 0.010767 | inf | 0.010658 | 1.01x |
| large_direct | 0.018666 | 0.016579 | inf | 0.013321 | 1.24x |
| persistent_forced | 0.018995 | 0.014564 | inf | 0.013623 | 1.07x |

Unit tests covered `tiny`, `default`, `large_direct`, and forced persistent dispatch.  All optimized outputs matched the analytic PyTorch/ACL reference with `max_abs=0`.

## cannsim setup

Sub-kernel hosts were built and executed with `cannsim_local_run(gen_report=True)`.

| Variant | Kernel | Elements in sub-kernel | Trace path |
|---|---|---:|---|
| Baseline | `_fill_ones_kernel`, `BLOCK_SIZE=128` | 128 | `/tmp/cannsim_local/l2_66_fill_baseline_v2/cannsim_20260630021434_test_kernel/report/trace_core0.json` |
| Optimized large path | `_fill_ones_direct`, `BLOCK_SIZE=1024` | 1024 | `/tmp/cannsim_local/l2_66_fill_optimized_v2/cannsim_20260630021610_test_kernel/report/trace_core0.json` |

Both cannsim runs reported chip cycle `419`; hardware latency is `419 cycles * 0.4 ns/cycle = 167.6 ns` for the one-program probe.  Normalized store throughput improves from `128/419 = 0.305 elements/cycle` to `1024/419 = 2.444 elements/cycle` for the large-tile path.

## cannsim trace pipeline table

Durations below are aggregate `ph=X` trace durations by category from `trace_core0.json`; percentages use trace span for the same core trace.

| Variant | Trace span | SCALAR | MTE3 | SCALARLDST | PUSHQ | RVECST | RVECEX | FLOWCTRL | MTE2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline BLOCK=128 | 5326 | 1732 (32.5%) | 868 (16.3%) | 479 (9.0%) | 391 (7.3%) | 18 (0.3%) | 12 (0.2%) | 9 (0.2%) | 5 (0.1%) |
| Optimized BLOCK=1024 | 5528 | 1739 (31.5%) | 1086 (19.6%) | 480 (8.7%) | 403 (7.3%) | 144 (2.6%) | 12 (0.2%) | 9 (0.2%) | 5 (0.1%) |

## Top latency events

| Variant | Event | Category | Duration |
|---|---|---:|---:|
| Baseline | `DC_PRELOAD_XN_IMM` | SCALAR | 493 |
| Baseline | `MOV_SRC_TO_DST_ALIGNv2` | MTE3 | 482 |
| Baseline | `LD_XD_XN_IMM` | SCALARLDST | 479 |
| Baseline | `PUSHQ VF` | PUSHQ | 387 |
| Baseline | `WAIT_FLAG_VEC` | MTE3 | 386 |
| Optimized | `MOV_SRC_TO_DST_ALIGNv2` | MTE3 | 688 |
| Optimized | `DC_PRELOAD_XN_IMM` | SCALAR | 494 |
| Optimized | `LD_XD_XN_IMM` | SCALARLDST | 480 |
| Optimized | `PUSHQ VF` | PUSHQ | 399 |
| Optimized | `WAIT_FLAG_VEC` | MTE3 | 398 |

## Interpretation

The one-program hardware cycle count is fixed by launch/prologue overhead for this tiny fill kernel, so the useful optimization is grid reduction and normalized throughput on larger batches.  The optimized path writes 8x more elements in the same reported chip cycles and improves remote large-direct wall time from `0.016579 ms` to `0.013321 ms`.
