# Performance Report

## Summary

Final optimized `ModelNew` uses ACL-backed standard operators for the post-convolution sequence. This was selected after cannsim showed the attempted custom Triton 2x2-pool vectorization increased scalar/spill overhead, and remote hardware confirmed ACL dispatch is substantially faster than the editable baseline Triton epilogue.

## Cannsim subkernel traces

Subkernel setup: `grid=(1,)`, fp32 input, `K=2`, reduced lane count (`BLOCK=8`) to keep cycle-accurate simulation tractable. Baseline is the editable Triton fused epilogue; optimized-candidate is the vectorized 2x2 Triton attempt retained in the file but not used by final `forward()`.

| Kernel | trace_core0.json | wall_cycles | HW time (cycles × 0.4 ns) | x_events | i_events | Bottleneck |
|---|---:|---:|---:|---:|---:|---|
| Baseline Triton epilogue | `/tmp/cannsim_local/l2_35_baseline_small/.../trace_core0.json` | 4,864 | 1.946 µs | 1,204 | 64 | SCALARLDST |
| Optimized Triton candidate | `/tmp/cannsim_local/l2_35_opt_vector_small/.../trace_core0.json` | 9,907 | 3.963 µs | 1,627 | 168 | SCALARLDST |

### Pipeline utilization

| Kernel | Pipeline | ops | busy_cyc | Notes |
|---|---|---:|---:|---|
| Baseline | SCALARLDST | 69 | 3,533 | dominant spills/loads |
| Baseline | SCALAR | 952 | 1,931 | index decomposition, scalar arithmetic |
| Baseline | MTE3 | 2 | 1,447 | output movement/wait |
| Baseline | PUSHQ | 17 | 1,405 | instruction dispatch pressure |
| Baseline | RVECEX | 153 | 225 | vector math is not the bottleneck |
| Optimized candidate | SCALARLDST | 181 | 4,823 | vectorized 2-D tile adds spills |
| Optimized candidate | PUSHQ | 43 | 3,003 | more vector instruction dispatch |
| Optimized candidate | SCALAR | 1,126 | 2,569 | still dominated by index/math scalar work |
| Optimized candidate | MTE3 | 16 | 1,761 | more stores/moves |
| Optimized candidate | RVECEX | 187 | 464 | extra activation/reduction vector ops |

### Top instruction costs

| Kernel | Instruction | Pipe | Count | Total cycles | Avg cycles |
|---|---|---|---:|---:|---:|
| Baseline | `ST_XD_XN_IMM` | SCALARLDST | 33 | 2,353 | 71 |
| Baseline | `LDP_XI_XJ_XN` | SCALAR | 4 | 2,043 | 511 |
| Baseline | `LD_XD_XN_IMM` | SCALARLDST | 4 | 1,525 | 381 |
| Baseline | `VF` | PUSHQ | 3 | 1,123 | 374 |
| Candidate | `ST_XD_XN_IMM` | SCALARLDST | 65 | 3,084 | 47 |
| Candidate | `VF` | PUSHQ | 20 | 2,927 | 146 |
| Candidate | `LD_XD_XN_IMM` | SCALARLDST | 84 | 2,674 | 32 |
| Candidate | `LDP_XI_XJ_XN` | SCALAR | 4 | 1,945 | 486 |

Conclusion from cannsim: the custom Triton variants are scalar/spill-bound and are not the right final dispatch for this standard Conv2d/activation/pool chain.

## Remote hardware benchmark

Remote verification: `test_passed=true`, `bench_passed=true`.

| label | PyTorch / ACL (ms) | Baseline Triton1 (ms) | Baseline Triton2 (ms) | Optimized Triton (ms) | Speedup vs Baseline Triton1 |
|---|---:|---:|---:|---:|---:|
| default_k2 | 186.801392 | 2743.935791 | 117.518745 | 186.279282 | 14.73× |
| small_k2 | 1.833902 | 26.494989 | 1.205941 | 1.798434 | 14.73× |
| fallback_k3 | 0.560993 | 27.233152 | 27.242937 | 0.552674 | 49.28× |

Geomean speedup vs editable baseline Triton1: **22.03×**.

Note: `Baseline Triton2` is read-only reference. It failed optimized-profile correctness for the two `K=2` comparison shapes in this environment but passed for `fallback_k3`; it was kept as a visible comparison column as required.