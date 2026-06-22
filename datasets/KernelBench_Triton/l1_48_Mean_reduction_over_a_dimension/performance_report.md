# Performance Report

## Cannsim setup

- Baseline trace: `/tmp/cannsim_local/l1_48_mean_baseline_small/cannsim_20260625015138_test_kernel/report/trace_core0.json`
- Optimized trace: `/tmp/cannsim_local/l1_48_mean_opt_small/cannsim_20260625015328_test_kernel/report/trace_core0.json`
- Sub-kernel: `dim=1`, `B=1`, `M=32`, `N=64`, grid `(1,)`; both host checks printed `[HOST] PASS`.
- Cycle-to-time conversion: `cycles * 0.4 ns`.

## Cannsim trace comparison

| Kernel | Wall cycles | Est. time | X events | I events | Bottleneck |
|---|---:|---:|---:|---:|---|
| Baseline dim=1 tiled | 7,663 | 3.065 µs | 1,245 | 102 | MTE2, 5,479 busy cycles |
| Optimized dim=1 block | 3,519 | 1.408 µs | 505 | 35 | SCALARLDST, 2,209 busy cycles |

## Pipeline utilization

| Pipeline | Baseline busy cycles | Optimized busy cycles | Change |
|---|---:|---:|---:|
| MTE2 | 5,479 | 1,022 | -81.3% |
| VEC | 5,410 | 1,015 | -81.2% |
| SCALARLDST | 1,621 | 2,209 | +36.3% |
| SCALAR | 1,485 | 889 | -40.1% |
| MTE3 | 1,436 | 1,252 | -12.8% |
| PUSHQ | 1,327 | 810 | -39.0% |
| RVECEX | 181 | 130 | -28.2% |
| RVECLD | 97 | 52 | -46.4% |
| RVECST | 90 | 27 | -70.0% |
| FLOWCTRL | 7 | 7 | 0.0% |

## Top instruction changes

| Instruction / event | Baseline | Optimized | Note |
|---|---:|---:|---|
| `MOV_SRC_TO_DST_ALIGNv2` total cycles | 16,267 | 1,013 | Blocked M reduction greatly reduces MTE2 transfer events. |
| `WAIT_FLAG_MTE2` total cycles | 9,645 | 1,013 | Fewer MTE waits after loading `[M,N]` blocks. |
| `JUMPC` count | 76 | 16 | `tl.range(0, M, BLOCK_M)` cuts loop-control pressure. |
| Wall cycles | 7,663 | 3,519 | 2.18x cannsim sub-kernel speedup. |

## Hardware latency

Remote NPU verification passed (`UNIT_TEST PASS`, `bench_passed=true`). Hardware benchmark from `profile_kernels.py`:

| Label | PyTorch / ACL (ms) | Baseline Triton1 (ms) | Baseline Triton2 (ms) | Optimized Triton (ms) | Speedup vs Baseline Triton1 |
|---|---:|---:|---:|---:|---:|
| small_dim1 | 0.004785 | 0.046067 | 0.066714 | 0.010843 | 4.25x |
| dim0_boundary | 0.003955 | inf | inf | 0.003995 | n/a |
| dim2_small | 0.004509 | inf | inf | 0.004516 | n/a |
| target_dim1 | 6.668901 | 20.419056 | 6.845290 | 7.951386 | 2.57x |

Target hardware latency: optimized `target_dim1` = **7.951386 ms**, baseline input kernel = **20.419056 ms**. Baseline2 is a read-only reference comparison and is faster on target (6.845290 ms); it was not modified.
