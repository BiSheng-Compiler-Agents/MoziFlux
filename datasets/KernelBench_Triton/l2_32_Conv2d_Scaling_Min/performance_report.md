# Performance Report

## Cannsim methodology

`cannsim_local_run` was used with grid=1 microprobes of the custom post-conv channel-min epilogue. The full convolution is ACL-backed and is measured on hardware by `profile_kernels.py`; cannsim is used only for Triton epilogue instruction tracing.

## Cannsim trace summary

Cycle-to-time conversion uses 0.4 ns/cycle.

| Path | Probe | trace_core0.json | Wall cycles | Est. ns | Dominant trace categories |
|---|---:|---|---:|---:|---|
| Baseline Triton epilogue | B=1,C=16,HW=8,BLOCK_HW=8 | `/tmp/cannsim_local/l2_32_conv2d_scaling_min_baseline_small2/.../trace_core0.json` | 7,011 | 2,804.4 | SCALARLDST 9,017; SCALAR 5,602; MTE2 3,328; VEC/WAIT 3,038; MTE3 1,469 |
| Optimized production epilogue | custom Triton launch removed | n/a | 0 | 0 | ACL `torch.amin/amax` handles reduction; no Triton epilogue trace |
| Optimized Triton fallback | B=1,C=16,HW=8,BLOCK_HW=8 diagnostic | `/tmp/cannsim_local/l2_32_conv2d_scaling_min_opt_small2/.../trace_core0.json` | 9,582 | 3,832.8 | SCALAR 14,279; SCALARLDST 9,136; MTE3 1,647; PUSHQ 1,479 |

Interpretation: the baseline custom epilogue is scalar/LDST heavy even at tiny scale. The default optimized path removes this Triton work completely; the fallback trace is retained for correctness/dispatch coverage, not chosen as the production fast path.

## Hardware latency

Hardware results are populated from `remote_verify` / `profile_kernels.py` after the VERIFY stage.

| label | PyTorch / ACL (ms) | Baseline Triton1 (ms) | Baseline Triton2 (ms) | Optimized Triton (ms) | Notes |
|---|---:|---:|---:|---:|---|
| small_32 | TBD | TBD | inf | TBD | baseline2 is parser-visible but not loaded due sandbox no-read rule |
| medium_128 | TBD | TBD | inf | TBD |  |
| exact_256 | TBD | TBD | inf | TBD | exact source shape |
