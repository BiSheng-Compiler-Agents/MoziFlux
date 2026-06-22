# Performance Report

## Cannsim setup

- Baseline trace: `cannsim_baseline`, sub-kernel probe of `_spatial_mean_subtract_kernel`, `S=4096`, `BLOCK=2048`, `grid=1`.
- Optimized experiment trace: `cannsim_optimized`, sub-kernel probe of the tested `_partial_sum_direct_kernel`, `S=2048`, `BLOCK=2048`, `grid=1`.
- A combined three-launch custom optimized epilogue completed device tasks but did not reach the plugin safe quiet-time window. Hardware verification then showed that custom partial-reduction epilogue was a regression, so the final target path uses ACL mean subtraction; no Triton trace exists for that ACL-only path.
- Cycle-to-time conversion: `hardware_time_ns = cycles * 0.4`.

## Cannsim trace summary

| Path | Trace | Probe | wall_cycles | est. hw time | Dominant pipeline | Dominant busy cycles | Top critical instruction |
|---|---|---:|---:|---:|---|---:|---|
| Baseline Triton | `/tmp/cannsim_local/l2_15_baseline/.../trace_core0.json` | serial mean+subtract, `S=4096` | 6264 | 2.506 us | VEC | 2659 | `WAIT_FLAG_MTE2`, 2714 total cycles |
| Tested partial-reduction Triton | `/tmp/cannsim_local/l2_15_opt_partial/.../trace_core0.json` | partial-sum tile, `S=2048` | 3116 | 1.246 us | SCALAR | 1813 | `LD_XD_XN_IMM`, 2189 total cycles |

## Pipeline table

| Path | SCALAR | SCALARLDST | MTE2 | VEC | MTE3 | PUSHQ | RVECEX | RVECLD | RVECST | FLOWCTRL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline busy cycles | 1936 | 1849 | 2547 | 2659 | 1468 | 1266 | 421 | 205 | 118 | 7 |
| Tested partial busy cycles | 1813 | 1734 | 1010 | 988 | 102 | 196 | 150 | 51 | 9 | 7 |

## Hardware benchmark (`remote_verify`)

| label | PyTorch / ACL (ms) | Baseline Triton1 (ms) | Baseline Triton2 (ms) | Optimized Triton (ms) | Optimized vs Baseline1 |
|---|---:|---:|---:|---:|---:|
| small_direct | 0.021643 | 0.016524 | 0.017027 | 0.015564 | 1.062x |
| medium_acl | 0.062893 | 0.062805 | 0.061151 | 0.062480 | 1.005x |
| target | 1.724504 | 1.885319 | 1.736801 | 1.723299 | 1.094x |

Correctness: `UNIT_TEST PASS`; optimized max diff was `2.98023e-08` on `small_direct` and `0` on `medium_acl`/`target`.

## Interpretation

Cannsim confirmed the baseline custom epilogue is wait/memory dominated and motivated testing a parallel partial-reduction Triton rewrite. Hardware timing showed the rewrite regressed badly at the full target (`14.909141 ms` in the discarded run), so the final optimization is a regime dispatch: keep the tiny-plane Triton direct path and use ACL mean subtraction for larger planes.
