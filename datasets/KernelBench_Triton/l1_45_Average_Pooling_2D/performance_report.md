# Performance Report

## Cannsim setup

- Baseline trace: `/tmp/cannsim_local/kb45_baseline/cannsim_20260625005448_test_kernel/report/trace_core0.json`
- Optimized trace: `/tmp/cannsim_local/kb45_optimized_scalar/cannsim_20260625010221_test_kernel/report/trace_core0.json`
- Both traces simulate one valid 11x11 output tile with `grid=(1,)`; persistent-grid dispatch savings are a full-shape hardware effect and are not visible in a single-program trace.
- Cycle-to-time conversion uses `0.4 ns/cycle`.

## Trace summary

| Kernel | Trace busy cycles | Est. trace time | Notes |
|---|---:|---:|---|
| Baseline Triton1 | 36904 | 14761.6 ns | Branch/count/bounds in pooling loop |
| Optimized Triton | 30533 | 12213.2 ns | No-padding scalar fast path |
| Delta | -6371 | -2548.4 ns | -17.3% trace-cycle change |

### Baseline by pipeline
| Pipeline | Busy cycles | Share | Est. time |
|---|---:|---:|---:|
| 01_SCALAR | 15452 | 41.9% | 6180.8 ns |
| 02_SCALARLDST | 8573 | 23.2% | 3429.2 ns |
| 10_PUSHQ | 7444 | 20.2% | 2977.6 ns |
| 13_RVECLD | 2280 | 6.2% | 912.0 ns |
| 12_RVECEX | 1560 | 4.2% | 624.0 ns |
| 14_RVECST | 1080 | 2.9% | 432.0 ns |
| 07_MTE3 | 501 | 1.4% | 200.4 ns |
| 15_FLOWCTRL | 9 | 0.0% | 3.6 ns |

### Optimized by pipeline
| Pipeline | Busy cycles | Share | Est. time |
|---|---:|---:|---:|
| 02_SCALARLDST | 9665 | 31.7% | 3866.0 ns |
| 01_SCALAR | 8149 | 26.7% | 3259.6 ns |
| 10_PUSHQ | 7284 | 23.9% | 2913.6 ns |
| 13_RVECLD | 2280 | 7.5% | 912.0 ns |
| 12_RVECEX | 1560 | 5.1% | 624.0 ns |
| 14_RVECST | 1080 | 3.5% | 432.0 ns |
| 07_MTE3 | 501 | 1.6% | 200.4 ns |
| 15_FLOWCTRL | 9 | 0.0% | 3.6 ns |

## Top instruction groups

| Kernel | Top groups |
|---|---|
| Baseline | VF:6964, LDP_XI_XJ_XN:3964, ST_XD_XN_IMM:3167, LD_XD_XN_IMM:2785, LD_XD_XN:2621, RV_VLDI:2280 |
| Optimized | VF:6804, LD_XD_XN_IMM:3262, LD_XD_XN:3254, ST_XD_XN_IMM:3149, LDP_XI_XJ_XN:2506, RV_VLDI:2280 |

## Hardware latency

`remote_verify` PASS on physical Ascend NPU. Latest benchmark table from `results.txt`:

| label | PyTorch / ACL (ms) | Baseline Triton1 (ms) | Baseline Triton2 (ms) | Optimized Triton (ms) |
|---|---:|---:|---:|---:|
| direct_small | 0.003766 | 0.091941 | 0.091965 | 0.090108 |
| persistent_mid | 0.050682 | inf (coreDim guard) | inf (coreDim guard) | 13.072899 |

The full source target (`16x64x2048x2048`) exceeds the baseline launch cap (`35,436,544` programs) and also failed the remote verifier during the oversized optimized run, so hardware timing is reported on the bounded persistent dispatch shape that exercises the same grid-capped path without poisoning the NPU context.
