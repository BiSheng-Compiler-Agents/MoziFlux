# Performance Report

## cannsim setup

- Tool: `cannsim_local_run(gen_report=True)`
- Sub-kernel host: grid=(1,1,1), M=128, N=128, K=64, fp32 inputs, all-one correctness check
- Trace files:
  - Baseline: `/tmp/cannsim_local/l1_6_baseline_v3/cannsim_20260622181646_test_kernel/report/trace_core0.json`
  - Optimized direct kernel: `/tmp/cannsim_local/l1_6_optimized_v1/cannsim_20260622181924_test_kernel/report/trace_core0.json`
- Hardware time conversion: cycles × 0.4 ns

## cannsim trace comparison

| Kernel | wall_cycles | hw latency (ns) | x_events | i_events | Bottleneck |
|---|---:|---:|---:|---:|---|
| Baseline direct | 12,999 | 5,199.6 | 5,754 | 210 | FIXP 5,411 busy cycles |
| Optimized direct | 13,580 | 5,432.0 | 5,880 | 215 | FIXP 4,839 busy cycles |

Single-core direct cannsim shows a 4.47% direct-path regression; the full-shape optimization benefit comes from split-K inter-core parallelism, which is intentionally not visible in a grid=(1,1,1) sub-kernel trace.

## Pipeline table

| Pipeline | Baseline busy_cyc | Optimized busy_cyc | Notes |
|---|---:|---:|---|
| FIXP | 5,411 | 4,839 | lower wait on optimized direct |
| FLOWCTRL | 4,942 | 4,715 | lower loop/control cost |
| MTE3 | 4,664 | 4,454 | lower store movement |
| CUBE | 4,448 | 4,446 | same one-tile MMAD work |
| MTE1 | 4,453 | 4,109 | lower cube wait |
| SCALARLDST | 2,664 | 3,563 | increased scalar load/store pressure |
| PUSHQ | 3,449 | 3,449 | unchanged |
| RVECST | 3,347 | 3,347 | unchanged for fp32 direct tile |
| RVECLD | 3,056 | 3,056 | unchanged for fp32 direct tile |
| SCALAR | 1,687 | 2,681 | increased scalar instructions from hints/addressing |
| MTE2 | 1,834 | 1,806 | similar load movement |
| VEC | 1,738 | 1,718 | similar vector wait |

## Hardware verification and latency

Remote Ascend hardware verification passed all unit tests:

```text
TEST baseline  direct_small    : PASS max_abs=0
TEST optimized direct_small    : PASS max_abs=0
TEST baseline  direct_boundary : PASS max_abs=0
TEST optimized direct_boundary : PASS max_abs=0
TEST baseline  split_path      : PASS max_abs=4.90341e-06
TEST optimized split_path      : PASS max_abs=4.88106e-06
```

Benchmark output:

| label | PyTorch / ACL (ms) | Baseline Triton (ms) | Optimized Triton (ms) | Opt vs baseline |
|---|---:|---:|---:|---:|
| direct_small | 0.004386 | 0.006639 | 0.006414 | 1.04× |
| direct_nonpow2 | 0.003625 | 0.004755 | 0.005699 | 0.83× |
| split_medium | 0.034103 | 0.082028 | 0.019935 | 4.11× |
| benchmark_largeK | 3.909485 | 23.173428 | 14.496786 | 1.60× |

The optimized kernel is 1.60× faster than the baseline Triton kernel on the target benchmark shape, but remains slower than PyTorch / ACL on that large-K case.
