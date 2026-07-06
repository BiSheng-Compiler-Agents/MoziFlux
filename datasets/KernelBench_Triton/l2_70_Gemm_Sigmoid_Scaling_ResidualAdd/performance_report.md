# Performance Report

## Method

cannsim was run with a one-program sub-kernel host for the post-GEMM epilogue `y = x + scale * sigmoid(x)`. Both baseline and optimized traces use `BLOCK_SIZE=16384`, `n_elements=16384`, `scale=2.0`, `grid=(1,1,1)`, and fp32 buffers. Hardware-time conversion uses `0.4 ns/cycle`.

Trace files:
- Baseline: `/tmp/cannsim_local/l2_70_baseline_b16384_full/cannsim_20260630044502_test_kernel/report/trace_core0.json`
- Optimized: `/tmp/cannsim_local/l2_70_opt_b16384_full/cannsim_20260630044653_test_kernel/report/trace_core0.json`

## cannsim latency summary

| Provider | BLOCK_SIZE | Elements | Wall cycles | Hardware latency (ns) | Cycles/element | Δ vs baseline |
|---|---:|---:|---:|---:|---:|---:|
| Baseline Triton epilogue | 16384 | 16384 | 5083 | 2033.2 | 0.31024 | 1.000x |
| Optimized Triton epilogue | 16384 | 16384 | 5086 | 2034.4 | 0.31042 | 0.999x |

## Pipeline busy-cycle table

Busy cycles overlap across pipelines, so percentages can sum above 100%.

| Pipeline | Baseline busy cycles | Baseline % wall | Optimized busy cycles | Optimized % wall | Direction |
|---|---:|---:|---:|---:|---:|
| RVECEX | 14092 | 277.2% | 14092 | 277.1% | same |
| SCALAR | 3475 | 68.4% | 3495 | 68.7% | +0.6% |
| MTE3 | 3304 | 65.0% | 3293 | 64.7% | -0.3% |
| MTE2 | 2511 | 49.4% | 2495 | 49.1% | -0.6% |
| RVECST | 2305 | 45.3% | 2305 | 45.3% | same |
| RVECLD | 2304 | 45.3% | 2304 | 45.3% | same |
| SCALARLDST | 1739 | 34.2% | 1753 | 34.5% | +0.8% |
| VEC | 1261 | 24.8% | 1253 | 24.6% | -0.6% |
| PUSHQ | 1119 | 22.0% | 1119 | 22.0% | same |
| FLOWCTRL | 9 | 0.2% | 9 | 0.2% | same |

## Top instructions

| Provider | Top instructions by busy cycles |
|---|---|
| Baseline | `RV_VDIV=4352`, `RV_VEXP=4096`, `WAIT_FLAG_VEC=2365`, `RV_VSTI=2305`, `RV_VLDI=2304`, `MOV_SRC_TO_DST_ALIGNv2=2199` |
| Optimized | `RV_VDIV=4352`, `RV_VEXP=4096`, `WAIT_FLAG_VEC=2357`, `RV_VSTI=2305`, `RV_VLDI=2304`, `MOV_SRC_TO_DST_ALIGNv2=2188` |

## Interpretation

The default epilogue is dominated by vector `exp`/`div` work from sigmoid. Removing cache/eviction hints and adding grid-cap-safe dispatch improves robustness/generalization, but the cannsim sub-kernel latency is effectively unchanged at the default 16384-element tile.

A tested 4096-element tile was rejected: it reduced raw wall cycles to 3585 but regressed normalized throughput to `0.875 cycles/element` versus `0.310 cycles/element` at `BLOCK_SIZE=16384`.

## Hardware latency

`remote_verify(run_test=True, run_bench=True)` passed correctness and benchmark on physical Ascend hardware.

| Shape | PyTorch / ACL (ms) | Baseline Triton1 (ms) | Baseline Triton2 (ms) | Optimized Triton (ms) | Optimized vs Baseline1 |
|---|---:|---:|---:|---:|---:|
| small_irregular | 0.096954 | 0.058992 | inf (sandbox skip) | 0.069130 | 0.853x |
| medium_aligned | 0.178521 | 0.086831 | inf (sandbox skip) | 0.154314 | 0.563x |
| default | 16.090981 | 15.358143 | inf (sandbox skip) | 15.368969 | 0.999x |

Correctness: Baseline Triton1 and Optimized Triton passed all benchmark shapes with `max_abs=0`; the forced persistent optimized path also passed with `max_abs=0`. Baseline Triton2 was kept parser-visible but skipped because the sandbox explicitly forbids reading `base_*.py`.
