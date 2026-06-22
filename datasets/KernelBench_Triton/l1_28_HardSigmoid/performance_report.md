# Performance Report: HardSigmoid

## Cannsim setup

- Baseline sub-kernel: `_hardsigmoid_kernel`, `BLOCK_SIZE=2048`, `grid=(1,)`, fp32, `N=2048`.
- Optimized sub-kernel: `_hardsigmoid_persistent_kernel`, `BLOCK_SIZE=8192`, `grid=(1,)`, `n_programs=1`, fp32, `N=8192`.
- Trace files:
  - Baseline: `/tmp/cannsim_local/l1_28_hardsigmoid_baseline2/cannsim_20260624174813_test_kernel/report/trace_core0.json`
  - Optimized: `/tmp/cannsim_local/l1_28_hardsigmoid_optimized/cannsim_20260624174959_test_kernel/report/trace_core0.json`
- Cycle conversion: `hardware_time_ns = cycles * 0.4`.

## Cannsim trace summary

| Kernel | Elements traced | Wall cycles | HW time | Normalized cycles / 2048 elts | Normalized HW time / 2048 elts | Bottleneck |
|---|---:|---:|---:|---:|---:|---|
| Baseline direct | 2048 | 3307 | 1.323 us | 3307 | 1.323 us | SCALAR |
| Optimized persistent | 8192 | 4277 | 1.711 us | 1069.25 | 0.428 us | MTE3 `WAIT_FLAG_VEC` |

Normalized cannsim improvement: `3307 / 1069.25 = 3.09x` per 2048 elements.

## Pipeline utilization

| Kernel | SCALAR busy | MTE3 busy | SCALARLDST busy | MTE2 busy | VEC busy | PUSHQ busy | RVECEX busy | RVECLD/RVECST busy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline direct | 1797 | 1508 | 1232 | 992 | 986 | 165 | 120 | 89 / 99 |
| Optimized persistent | 1790 | 2151 | 1692 | 1091 | 1087 | 452 | 407 | 378 / 386 |

## Top trace instructions

| Kernel | Critical instructions |
|---|---|
| Baseline direct | `LDP_XI_XJ_XN` 1456 cyc, `LD_XD_XN_IMM` 1232 cyc, `STI_XN_IMM` 1231 cyc, `WAIT_FLAG_VEC` 1135 cyc, `WAIT_FLAG_MTE2` 986 cyc |
| Optimized persistent | `LD_XD_XN_IMM` 1692 cyc, `RV_VSEL` 1536 cyc, `WAIT_FLAG_VEC` 1531 cyc, `STI_XN_IMM` 1215 cyc, `RV_VSTI` 1192 cyc, `RV_VLDI` 1152 cyc |

## Hardware benchmark (`remote_verify`)

Correctness passed for optimized on every benchmark shape. `Baseline Triton1` is skipped on the original shape because its direct grid would exceed `65535` programs.

| Shape label | PyTorch / ACL (ms) | Baseline Triton1 (ms) | Baseline Triton2 (ms) | Optimized Triton (ms) |
|---|---:|---:|---:|---:|
| direct_1M | 0.009409 | 0.013512 | 0.013175 | 0.010517 |
| direct_irregular | 0.009458 | 0.013611 | 0.012859 | 0.010662 |
| persistent_original | 8.921406 | inf | 9.953013 | 9.190384 |

Hardware ratios:
- Direct 1M: optimized is `1.28x` faster than Baseline Triton1.
- Direct irregular: optimized is `1.28x` faster than Baseline Triton1.
- Original persistent shape: optimized is `1.08x` faster than Baseline Triton2 and is launchable where Baseline Triton1 overflows the grid cap.
