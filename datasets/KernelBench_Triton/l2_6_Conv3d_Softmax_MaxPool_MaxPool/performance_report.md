# Performance Report

## Cannsim setup

- Baseline trace: `/tmp/cannsim_local/l2_6_conv3d_softmax_pool_baseline_stride/cannsim_20260630032402_test_kernel/report/trace_core0.json`
- Optimized trace: `/tmp/cannsim_local/l2_6_conv3d_softmax_pool_opt_stride/cannsim_20260630032545_test_kernel/report/trace_core0.json`
- Microprobe: one output-width tile, `C=16`, production-like post-conv strides (`D=14,H=30,W=30`), `K=1` to keep cycle-accurate simulation tractable while preserving the channel-addressing pattern. Full `K=4` pooling is covered by the Python/NPU profiler.
- Cycle-to-time conversion: `hardware_time_ns = cycles * 0.4`.

## Cannsim trace summary

| Metric | Baseline Triton1 | Optimized Triton | Delta |
|---|---:|---:|---:|
| trace events | 513 | 503 | -1.9% |
| wall cycles | 3,200 | 3,299 | +3.1% |
| hardware time (ns) | 1,280.0 | 1,319.6 | +3.1% |
| summed busy cycles | 16,056 | 13,510 | -15.9% |
| host correctness | PASS, maxdiff=7.45e-09 | PASS, maxdiff=7.45e-09 | - |

## Pipeline breakdown by category

| Category | Baseline cycles | Baseline % | Optimized cycles | Optimized % | Delta |
|---|---:|---:|---:|---:|---:|
| SCALAR | 4,308 | 26.8% | 2,050 | 15.2% | -52.4% |
| SCALARLDST | 4,214 | 26.2% | 2,924 | 21.6% | -30.6% |
| MTE2 | 1,807 | 11.3% | 1,937 | 14.3% | +7.2% |
| RVECEX | 1,606 | 10.0% | 1,645 | 12.2% | +2.4% |
| MTE3 | 1,348 | 8.4% | 1,481 | 11.0% | +9.9% |
| VEC | 911 | 5.7% | 954 | 7.1% | +4.7% |
| PUSHQ | 842 | 5.2% | 1,059 | 7.8% | +25.8% |
| RVECLD | 703 | 4.4% | 989 | 7.3% | +40.7% |
| RVECST | 306 | 1.9% | 444 | 3.3% | +45.1% |

## Top instructions

| Rank | Baseline instruction | Cycles | Count | Optimized instruction | Cycles | Count |
|---:|---|---:|---:|---|---:|---:|
| 1 | LDP_XI_XJ_XN | 2,522 | 5 | LD_XD_XN_IMM | 1,702 | 2 |
| 2 | ST_XD_XN_IMM | 2,460 | 2 | WAIT_FLAG_VEC | 1,349 | 1 |
| 3 | LD_XD_XN_IMM | 1,754 | 3 | ST_XD_XN_IMM | 1,222 | 1 |
| 4 | WAIT_FLAG_VEC | 1,222 | 1 | VF | 1,047 | 3 |
| 5 | MOV_SRC_TO_DST_ALIGNv2 | 1,045 | 2 | MOV_SPR_XN | 975 | 9 |

## Interpretation

The channel-last kernel reduces scalar and scalar-load/store busy cycles substantially, confirming that the baseline's strided-channel addressing was a major local bottleneck. The sub-kernel wall time is effectively wait/MTE limited and is slightly higher for the optimized microprobe; final production latency must therefore be decided by the hardware profiler, where the full `K=4` pooling and materialization cost are included.

## Hardware latency

Remote hardware verification passed after bounding timeout-prone comparison providers. Hardware `@perf_report` latencies (ms):

| Shape | PyTorch / ACL | Baseline Triton1 | Baseline Triton2 | Optimized Triton |
|---|---:|---:|---:|---:|
| small_direct | 0.343644 | inf (pre-skipped) | inf (reference unavailable) | 0.344725 |
| generic_c24 | 0.810060 | inf (pre-skipped) | inf (reference unavailable) | 0.820889 |
| target | 11.270019 | inf (pre-skipped) | inf (reference unavailable) | 11.302918 |

Correctness: `UNIT_TEST PASS`; optimized max_abs was 0.0 on all benchmark shapes plus forced-persistent and wide-channel ACL fallback coverage.
