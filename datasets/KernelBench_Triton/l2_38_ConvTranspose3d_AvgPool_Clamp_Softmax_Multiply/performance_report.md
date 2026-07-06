# Performance Report

## Cannsim setup

- Baseline trace: `/tmp/cannsim_local/kb38_baseline/cannsim_20260629191549_test_kernel/report/trace_core0.json`
- Optimized trace: `/tmp/cannsim_local/kb38_optimized_final_direct64/cannsim_20260629192918_test_kernel/report/trace_core0.json`
- Probe shape: one NCDHW softmax tile, `N=1, C=64, DHW=64`, fp32.
- Hardware time conversion: `cycles * 0.4 ns`.

## Cannsim trace summary

| Kernel | wall_cycles | hardware time | x_events | i_events | bottleneck |
|---|---:|---:|---:|---:|---|
| Baseline Triton tiled | 4,854 | 1,941.6 ns | 2,419 | 41 | MTE3 (2,886 busy cycles) |
| Optimized Triton direct | 4,880 | 1,952.0 ns | 2,425 | 41 | MTE3 (2,904 busy cycles) |

The optimized direct Triton kernel has essentially identical per-tile work to the baseline (`+0.5%` cycles); the target-shape improvement comes from host dispatch avoiding the baseline's oversized 65,536-tile launch and using CANN's native clamp/softmax path.

## Pipeline tables

### Baseline

| Pipeline | ops | busy_cyc | lane_sum | lanes | window |
|---|---:|---:|---:|---:|---|
| MTE3 | 2 | 2,886 | 2,886 | 1 | [5827,8714] |
| SCALAR | 255 | 1,969 | 5,851 | 11 | [3869,8719] |
| PUSHQ | 11 | 1,966 | 1,997 | 2 | [4441,8271] |
| SCALARLDST | 9 | 1,833 | 2,335 | 3 | [3889,5854] |
| RVECEX | 1,552 | 1,295 | 10,475 | 16 | [4781,8234] |
| RVECLD | 322 | 1,211 | 2,968 | 18 | [6776,8229] |
| MTE2 | 3 | 1,018 | 1,938 | 2 | [5705,6751] |
| VEC | 1 | 1,011 | 1,011 | 1 | [5739,6750] |
| RVECST | 258 | 684 | 2,322 | 9 | [4787,8259] |

Top baseline instructions: `RV_VLDI` 2,968 cycles, `WAIT_FLAG_VEC` 2,445 cycles, `RV_VSTI` 2,322 cycles, `VF` 1,880 cycles.

### Optimized direct Triton path

| Pipeline | ops | busy_cyc | lane_sum | lanes | window |
|---|---:|---:|---:|---:|---|
| MTE3 | 2 | 2,904 | 2,904 | 1 | [5850,8755] |
| SCALAR | 261 | 1,981 | 5,723 | 10 | [3884,8760] |
| PUSHQ | 11 | 1,964 | 1,995 | 2 | [4451,8295] |
| SCALARLDST | 9 | 1,829 | 2,331 | 3 | [3904,5877] |
| RVECEX | 1,552 | 1,295 | 10,475 | 16 | [4789,8258] |
| RVECLD | 322 | 1,211 | 2,968 | 18 | [6800,8253] |
| MTE2 | 3 | 1,020 | 1,941 | 2 | [5728,6775] |
| VEC | 1 | 1,013 | 1,013 | 1 | [5761,6774] |
| RVECST | 258 | 684 | 2,322 | 9 | [4795,8283] |

Top optimized instructions: `RV_VLDI` 2,968 cycles, `WAIT_FLAG_VEC` 2,446 cycles, `RV_VSTI` 2,322 cycles, `VF` 1,878 cycles.

## Remote hardware benchmark

Remote verification passed correctness and benchmark.

| label | PyTorch / ACL (ms) | Baseline Triton1 (ms) | Baseline Triton2 (ms) | Optimized Triton (ms) |
|---|---:|---:|---:|---:|
| direct_small | 0.431836 | 0.359965 | 0.359437 | 0.359187 |
| direct_medium | 1.806396 | 1.697713 | 1.698720 | 1.694708 |
| target_acl | 540.028137 | inf (grid-cap preskip) | 625.447144 | 538.535034 |

Speedups:

| comparison | speedup |
|---|---:|
| direct_small vs Baseline Triton1 | 1.0022x |
| direct_medium vs Baseline Triton1 | 1.0018x |
| target_acl vs Baseline Triton2 | 1.1614x |
| target_acl vs PyTorch / ACL | 1.0028x |

Correctness: optimized `max_diff=2.23517e-08` on direct_small, direct_medium, and target_acl.
