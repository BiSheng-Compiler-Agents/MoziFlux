# Performance Report

## Cannsim sub-kernel traces

Cannsim was run with `cannsim_local_run(..., gen_report=True)` using sub-kernel hosts inside this workspace:

- Baseline: `/tmp/cannsim_local/l2_55_baseline_small/cannsim_20260629232646_test_kernel/report/trace_core0.json`
- Optimized fallback: `/tmp/cannsim_local/l2_55_opt/cannsim_20260629232841_test_kernel/report/trace_core0.json`

The baseline microprobe was intentionally shrunk to one row, one pool pair, and `IN=16` because the original vector matmul body timed out at larger sub-kernel size.  The optimized fallback probe covers 16 rows × 16 output pairs × `K=64`; normalized cycles are therefore the meaningful comparison.

| Kernel trace | Work represented | wall cycles | ns @ 0.4 ns/cyc | Dominant pipeline | Notes |
|---|---:|---:|---:|---|---|
| Baseline vector fused | 32 MACs + max/sum | 6,692 | 2,676.8 | PUSHQ 2,948 cyc | Vector matmul via `tl.sum(w*x)`; heavy scalar/dispatch overhead. |
| Optimized Triton fallback | 32,768 MACs + pair max partials | 5,561 | 2,224.4 | PUSHQ 3,816 cyc | Cube path active; CUBE 589 cyc, MTE3 3,089 cyc. |

| Normalized metric | Baseline | Optimized fallback | Ratio |
|---|---:|---:|---:|
| cycles / MAC | 209.125 | 0.1697 | ~1232× lower |
| ns / MAC | 83.65 | 0.0679 | ~1232× lower |

## Cannsim pipeline breakdown

| Pipeline | Baseline busy cycles | Optimized busy cycles |
|---|---:|---:|
| PUSHQ | 2,948 | 3,816 |
| SCALARLDST | 2,596 | 2,456 |
| MTE2 | 1,863 | 934 |
| VEC | 1,775 | 900 |
| SCALAR | 1,009 | 1,092 |
| RVECEX | 282 | 1,810 |
| RVECLD | 226 | 1,355 |
| RVECST | 226 | 2,045 |
| CUBE | 0 | 589 |
| MTE3 | 111 | 3,089 |

## Hardware latency (`remote_verify`)

Correctness passed on optimized production and fallback paths.  `Baseline Triton2` (read-only `base_*.py`) failed with `MLIRCompilationError` and is reported as `inf`.

| label | PyTorch / ACL (ms) | Baseline Triton1 (ms) | Baseline Triton2 (ms) | Optimized Triton (ms) | Optimized TritonFallback (ms) |
|---|---:|---:|---:|---:|---:|
| small_fallback | 0.118540 | 1.025326 | inf | 0.122819 | 0.103357 |
| irregular_acl | 0.125548 | 1.560702 | inf | 0.121913 | 0.170513 |
| default_acl | 48.042503 | inf | inf | 47.546478 | inf |

Default-shape optimized/PyTorch ratio: `48.042503 / 47.546478 = 1.010×`.
