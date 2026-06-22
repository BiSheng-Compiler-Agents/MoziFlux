# Performance Report

## Cannsim setup

| Provider | Host | Simulated tile | Trace path |
|---|---:|---:|---|
| Baseline Triton1 | one row argmax | `K=4096`, 1 output | `/tmp/cannsim_local/l1_51_argmax_baseline2/cannsim_20260625023624_test_kernel/report/trace_core0.json` |
| Optimized Triton | one dim=1 tile | `M=64, N=128`, 128 outputs | `/tmp/cannsim_local/l1_51_argmax_opt_m64/cannsim_20260625024123_test_kernel/report/trace_core0.json` |

The optimized sub-kernel is intentionally smaller than the full target reduction to keep cannsim practical; it exercises the same `[BLOCK_M, BLOCK_N]` vector-reduction body and amortizes results over 128 output columns.

## Cannsim trace summary

| Metric | Baseline | Optimized | Notes |
|---|---:|---:|---|
| wall cycles | 7,046 | 6,902 | raw sub-kernel cycles |
| outputs per simulated program | 1 | 128 | optimized computes 128 columns |
| cycles / output | 7,046.0 | 53.9 | normalized by output count |
| hardware time / output @0.4 ns | 2,818.4 ns | 21.6 ns | cannsim cycle-to-time conversion |
| x_events | 1,720 | 2,842 | more vector work per program in optimized tile |
| i_events | 55 | 21 | fewer instant/control events |

## Pipeline utilization

| Pipeline | Baseline busy cycles | Optimized busy cycles | Interpretation |
|---|---:|---:|---|
| MTE2 | 2,936 | 1,067 | fewer GM load transactions per output after 2D tiling |
| VEC | 2,854 | 1,057 | fewer MTE wait cycles |
| PUSHQ | 1,680 | 5,336 | optimized pushes wider vector work for 128 outputs |
| RVECEX | 1,028 | 2,634 | more useful vector comparison/index work per program |
| RVECLD | 586 | 4,479 | index/vector local loads for 128-lane tile |
| MTE3 | 339 | 5,969 | int64 stores for 128 outputs dominate raw tile trace |
| SCALAR | 788 | 600 | lower scalar work despite more outputs |

## Top trace bottlenecks

| Provider | Bottleneck | Critical instructions |
|---|---|---|
| Baseline | MTE2 (2,936 cycles) | `MOV_SRC_TO_DST_ALIGNv2`, `WAIT_FLAG_MTE2`, `RV_VCMAX`, `RV_VCMIN` |
| Optimized | MTE3 (5,969 cycles raw) | `WAIT_FLAG_VEC`, `RV_VLDI`, `VF`, `RV_VSTI` |

## Hardware latency

Remote verification passed (`UNIT_TEST PASS`, `bench_passed=true`). Latencies from `results.txt`:

| label | PyTorch / ACL (ms) | Baseline Triton1 (ms) | Baseline Triton2 (ms) | Optimized Triton (ms) |
|---|---:|---:|---:|---:|
| small_dim1 | 0.006141 | 0.011735 | 0.005974 | 0.005305 |
| irregular_dim1 | 0.006512 | 0.016545 | 0.007080 | 0.021318 |
| target_dim1 | 5.435073 | inf | inf | 18.542612 |

Notes: the editable baseline overflows Ascend `coreDim` on the target (`B*N=524,160`) and the read-only baseline is skipped for target safety, so target speedup versus baseline is not finite. The optimized Triton path is correctness- and legality-improved for the target but remains slower than ACL (`5.435073 / 18.542612 = 0.293x`).
