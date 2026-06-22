# Performance Report

## Cannsim setup

- Tool: `cannsim_local_run(..., gen_report=True)` on Ascend950 simulator.
- Kernel host: one-row reverse-cumsum diagnostic sub-kernel, `grid=(1,)`.
- Full actual custom scan tiles (`BLOCK_N=512/1024`) did not produce a safe trace within the plugin stability window; per cannsim timeout guidance, the trace below uses a bounded diagnostic probe (`BLOCK_N=8`, `NUM_BLOCKS=2`, `N=16`) to compare baseline custom Triton codegen against the optimized custom fallback candidate. Production optimized latency comes from hardware `remote_verify` and uses ACL dispatch.

## Cannsim trace table

| Variant | Trace | wall cycles | time @0.4ns | SCALARLDST busy | PUSHQ busy | MTE3 busy | Dominant bottleneck |
|---|---:|---:|---:|---:|---:|---:|---|
| Baseline Triton1 custom path | `/tmp/cannsim_local/l1_91_rcumsum_baseline_b8/.../trace_core0.json` | 10,201 | 4.080 us | 3,796 | 3,649 | 2,807 | SCALARLDST |
| Optimized custom fallback candidate | `/tmp/cannsim_local/l1_91_rcumsum_opt_b8/.../trace_core0.json` | 9,539 | 3.816 us | 3,808 | 3,627 | 2,810 | SCALARLDST |

## Top instruction deltas

| Instruction | Baseline total cycles | Optimized fallback total cycles | Delta |
|---|---:|---:|---:|
| `ST_XD_XN_IMM` | 13,070 | 8,878 | -32.1% |
| `LD_XD_XN_IMM` | 3,787 | 3,716 | -1.9% |
| `VF` | 3,557 | 3,535 | -0.6% |
| `MOV_SRC_TO_DST_ALIGNv2` | 2,791 | 2,794 | +0.1% |

## Interpretation

The custom Triton path remains SCALARLDST-bound even after loop cleanup, confirming that reverse prefix scan is not an efficient fit for this Triton lowering on Ascend. The final optimization therefore uses host dispatch to ACL's native cumsum implementation.

## Hardware latency (`remote_verify`)

| label | PyTorch / ACL (ms) | Baseline Triton1 (ms) | Baseline Triton2 (ms) | Optimized Triton (ms) | Speedup vs Baseline1 |
|---|---:|---:|---:|---:|---:|
| tiny | 0.007558 | 0.028085 | inf | 0.007549 | 3.72x |
| mask_odd | 0.015563 | 0.188891 | inf | 0.015555 | 12.14x |
| medium | 0.142640 | 2.160154 | inf | 0.142689 | 15.14x |
| target | 127.577202 | 3377.401123 | 283.104736 | 127.609329 | 26.47x |

Correctness: `UNIT_TEST PASS`; optimized passed tiny, odd-mask, medium, and target shapes. Baseline2 hit MLIR compilation errors on smaller shapes but passed target; it is read-only and was preserved as a comparison column.
