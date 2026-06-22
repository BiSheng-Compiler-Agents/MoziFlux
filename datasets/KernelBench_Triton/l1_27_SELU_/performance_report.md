# Performance Report: SELU

## Cannsim setup

- Baseline sub-kernel: `_selu_kernel`, `BLOCK_SIZE=4096`, `grid=(1,)`, fp32, `N=4096`.
- Optimized sub-kernel: `_selu_persistent_kernel`, `BLOCK_SIZE=8192`, `grid=(1,)`, `n_programs=1`, fp32, `N=8192`.
- Trace files:
  - Baseline: `/tmp/cannsim_local/l1_27_selu_baseline/cannsim_20260624173636_test_kernel/report/trace_core0.json`
  - Optimized: `/tmp/cannsim_local/l1_27_selu_optimized/cannsim_20260624173823_test_kernel/report/trace_core0.json`
- Cycle conversion: `hardware_time_ns = cycles * 0.4`.

## Cannsim trace summary

| Kernel | Elements traced | Wall cycles | HW time | Normalized cycles / 4096 elts | Normalized HW time / 4096 elts | Bottleneck |
|---|---:|---:|---:|---:|---:|---|
| Baseline direct | 4096 | 3885 | 1.554 us | 3885 | 1.554 us | MTE3 `WAIT_FLAG_VEC` |
| Optimized persistent | 8192 | 5108 | 2.043 us | 2554 | 1.022 us | MTE3 `WAIT_FLAG_VEC` |

Normalized cannsim improvement: `3885 / 2554 = 1.52x` per element.

## Pipeline utilization

| Kernel | MTE3 busy | SCALAR busy | SCALARLDST busy | MTE2 busy | VEC busy | RVECEX busy | RVECLD/RVECST busy |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline direct | 2098 | 1785 | 1221 | 1009 | 1003 | 628 | 536 / 468 |
| Optimized persistent | 2966 | 1810 | 1707 | 1093 | 1089 | 1224 | 1091 / 905 |

## Top trace instructions

| Kernel | Critical instructions |
|---|---|
| Baseline direct | `WAIT_FLAG_VEC` 1655 cyc, `LD_XD_XN_IMM` 1221 cyc, `STI_XN_IMM` 1220 cyc, `WAIT_FLAG_MTE2` 1003 cyc |
| Optimized persistent | `WAIT_FLAG_VEC` 2349 cyc, `RV_VMULS` 2048 cyc, `RV_VEXP` 2048 cyc, `LD_XD_XN_IMM` 1707 cyc, `VF` 1265 cyc |

## Hardware benchmark (`remote_verify`)

Correctness passed for optimized on every benchmark shape. `Baseline Triton1` is intentionally skipped on the original shape because its direct grid would exceed `65535` programs.

| Shape label | PyTorch / ACL (ms) | Baseline Triton1 (ms) | Baseline Triton2 (ms) | Optimized Triton (ms) |
|---|---:|---:|---:|---:|
| direct_1M | 0.009588 | 0.011255 | 0.010692 | 0.010956 |
| direct_irregular | 0.009618 | 0.011295 | 0.010529 | 0.010990 |
| persistent_original | 8.923771 | inf | 10.038314 | 9.351640 |

Hardware ratios:
- Direct 1M: optimized is `1.03x` faster than Baseline Triton1.
- Direct irregular: optimized is `1.03x` faster than Baseline Triton1.
- Original persistent shape: optimized is `1.07x` faster than Baseline Triton2 and is launchable where Baseline Triton1 overflows the grid cap.
