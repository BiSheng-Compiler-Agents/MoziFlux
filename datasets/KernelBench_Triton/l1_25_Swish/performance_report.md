# Performance Report: Swish

## Cannsim setup

- Baseline kernel: `_swish_kernel`, `BLOCK_SIZE=4096`, sub-kernel `N=4096`, `grid=(1,)`
- Optimized kernel: `_swish_persistent_kernel`, `BLOCK_SIZE=8192`, sub-kernel `N=8192`, `n_programs=1`, `grid=(1,)`
- Trace files:
  - Baseline: `/tmp/cannsim_local/l1_25_swish_baseline/cannsim_20260624165945_test_kernel/report/trace_core0.json`
  - Optimized: `/tmp/cannsim_local/l1_25_swish_opt/cannsim_20260624170133_test_kernel/report/trace_core0.json`
- Cycle conversion: `hardware_time_ns = cycles * 0.4`

## Cannsim trace summary

| Kernel | Elements/tile | Wall cycles | Hardware ns | Cycles / 4096 elems | Hardware ns / 4096 elems | Dominant busy pipes |
|---|---:|---:|---:|---:|---:|---|
| Baseline | 4096 | 3589 | 1435.6 | 3589.0 | 1435.6 | SCALAR 3478, RVECEX 3078, MTE2 2014 |
| Optimized | 8192 | 4429 | 1771.6 | 2214.5 | 885.8 | RVECEX 6150, SCALAR 3552, MTE3 2300 |

Normalized to 4096 elements, the optimized tile is `3589 / 2214.5 = 1.62x` faster in cannsim. Full-shape dispatch also improves because the benchmark tile count changes from `393216` baseline direct programs to `65535` capped persistent programs.

## Cannsim detailed pipe table

| Kernel | SCALAR | RVECEX | MTE2 | MTE3 | SCALARLDST | VEC | RVECLD | RVECST | PUSHQ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline busy cycles | 3478 | 3078 | 2014 | 1803 | 1230 | 1011 | 576 | 576 | 346 |
| Optimized busy cycles | 3552 | 6150 | 2175 | 2300 | 1694 | 2176 | 1152 | 1152 | 602 |
| Optimized normalized / 4096 elems | 1776 | 3075 | 1088 | 1150 | 847 | 1088 | 576 | 576 | 301 |

The optimized trace doubles the tile size, so raw vector work roughly doubles where expected. Normalized memory and scalar overhead per 4096 elements drop, while RVECEX remains dominated by `sigmoid` (`RV_VEXP`/`RV_VDIV`), the true Swish math cost.

## Hardware latency (`remote_verify`)

Correctness: `UNIT_TEST PASS` for optimized direct and persistent paths.

| Shape label | Shape | PyTorch / ACL (ms) | Baseline Triton1 (ms) | Baseline Triton2 (ms) | Optimized Triton (ms) | Notes |
|---|---:|---:|---:|---:|---:|---|
| direct_small | 1024 x 1024 | 0.014102 | 0.010636 | 0.009728 | 0.010664 | Direct path |
| direct_medium | 4096 x 8192 | 0.509341 | 0.283147 | 0.238085 | 0.260082 | Direct path |
| persistent_original | 4096 x 393216 | 22.084715 | inf | inf | 9.252346 | Baselines skipped: `coreDim > 65535`; optimized persistent path |

For the required original shape, optimized Triton is `22.084715 / 9.252346 = 2.39x` faster than PyTorch / ACL and avoids the baseline Triton grid overflow.
