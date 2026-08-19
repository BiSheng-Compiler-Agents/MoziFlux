# Performance Report

## cannsim setup

- Baseline diagnostic kernel: flattened fp32 `ReLU(HardSwish)` formula matching `69_Conv2d_HardSwish_ReLU.py`.
- Optimized diagnostic kernel: algebraic piecewise formula from `opt_69_Conv2d_HardSwish_ReLU.py`.
- Sub-kernel shape: `N_ELEMS=4096`, `BLOCK_SIZE=4096`, `grid=(1,)`.
- Trace paths:
  - Baseline: `/tmp/cannsim_local/l2_69_hswish_baseline2/cannsim_20260630025514_test_kernel/report/trace_core0.json`
  - Optimized: `/tmp/cannsim_local/l2_69_hswish_optimized/cannsim_20260630025702_test_kernel/report/trace_core0.json`

## cannsim trace summary

Hardware latency estimate uses `cycles * 0.4 ns`.

| Kernel | Trace events | Wall cycles | HW latency (ns) | Speedup vs baseline |
|---|---:|---:|---:|---:|
| Baseline formula | 1335 | 4043 | 1617.2 | 1.000x |
| Optimized formula | 1012 | 3757 | 1502.8 | 1.076x |

## Pipeline busy-cycle table

Busy-cycle percentages can exceed 100% of wall cycles because trace lanes overlap.

| Pipeline | Baseline cycles | Baseline %wall | Optimized cycles | Optimized %wall | Change |
|---|---:|---:|---:|---:|---:|
| RVECEX | 7006 | 173.3% | 5022 | 133.7% | -28.3% |
| SCALAR | 3558 | 88.0% | 3504 | 93.3% | -1.5% |
| MTE2 | 2000 | 49.5% | 2007 | 53.4% | +0.4% |
| MTE3 | 2253 | 55.7% | 1977 | 52.6% | -12.3% |
| SCALARLDST | 1218 | 30.1% | 1220 | 32.5% | +0.2% |
| VEC | 1008 | 24.9% | 1011 | 26.9% | +0.3% |
| PUSHQ | 807 | 20.0% | 527 | 14.0% | -34.7% |
| RVECLD | 576 | 14.2% | 576 | 15.3% | 0.0% |
| RVECST | 576 | 14.2% | 584 | 15.5% | +1.4% |
| FLOWCTRL | 9 | 0.2% | 9 | 0.2% | 0.0% |

## Top instruction changes

| Rank | Baseline top instruction | Cycles | Optimized top instruction | Cycles |
|---:|---|---:|---|---:|
| 1 | MTE3 `WAIT_FLAG_VEC` | 1791 | MTE3 `WAIT_FLAG_VEC` | 1515 |
| 2 | SCALAR `LDP_XI_XJ_XN` | 1456 | SCALAR `LDP_XI_XJ_XN` | 1423 |
| 3 | RVECEX `RV_VADDS` | 1344 | SCALARLDST `LD_XD_XN_IMM` | 1220 |
| 4 | SCALARLDST `LD_XD_XN_IMM` | 1218 | SCALAR `STI_XN_IMM` | 1219 |
| 5 | SCALAR `STI_XN_IMM` | 1217 | VEC `WAIT_FLAG_MTE2` | 1011 |
| 6 | RVECEX `RV_VMINS` | 1152 | MTE2 `MOV_SRC_TO_DST_ALIGNv2` | 1010 |
| 7 | RVECEX `RV_VMAXS` | 1152 | MTE2 `MOV_SPR_XN` | 997 |
| 8 | VEC `WAIT_FLAG_MTE2` | 1008 | RVECEX `RV_VADDS` | 896 |

## Remote hardware benchmark

`profile_kernels.py` was run through `remote_verify` on Ascend NPU. `Baseline Triton2` is parser-visible but skipped because the sandbox forbids reading/importing `base_*.py`.

| label | PyTorch / ACL (ms) | Baseline Triton1 (ms) | Baseline Triton2 | Optimized Triton (ms) | Opt vs Baseline1 |
|---|---:|---:|---:|---:|---:|
| small_irregular | 0.285151 | 0.140805 | inf | 0.154349 | 0.912x |
| medium_rect | 1.157549 | 0.598359 | inf | 0.577327 | 1.036x |
| default | 62.840015 | 19.460703 | inf | 18.281796 | 1.064x |

## Correctness

All optimized paths passed:

```text
TEST Optimized Triton small_irregular: PASS max_abs=0
TEST Optimized Triton medium_rect: PASS max_abs=0
TEST Optimized Triton default: PASS max_abs=0
TEST Optimized Triton forced_persistent: PASS max_abs=0
UNIT_TEST PASS
```
