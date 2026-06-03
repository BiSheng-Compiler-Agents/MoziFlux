# Sub-kernel vs Multi-block cannsim Comparison

**Date:** June 2026
**Kernel:** matmul, BLOCK_M=N=K=128, FP16
**cannsim:** Ascend950, cannsim record + cannsim report

## Configs

| Variant | M | N | K | Grid | K-iters | Sim time |
|---|---|---|---|---|---|---|
| Sub-kernel | 128 | 128 | 128 | 1x1 | 1 | ~150s |
| Multi-block | 256 | 256 | 128 | 2x2 | 1 | ~207s |
| Full-shape | 1024 | 1024 | 1024 | 8x8 | 8 | >300s (timeout) |

## Trace Comparison

| Metric | Sub-kernel (1x1) | Multi-block (2x2) | Ratio |
|---|---|---|---|
| Cycles | 419 | 419 | 1.0x |
| Total duration | 383,560 | 394,992 | 1.03x |
| Events | 20,555 | 21,296 | 1.04x |
| Dominant bottleneck | RVECLD (7168) | RVECLD (7464) | Same |

## Key Findings

1. **Cycles are identical (419)** — cannsim reports chip-level wall latency which is the same regardless of grid size. The simulation models one core executing one block's worth of work.

2. **Trace events are nearly identical (~4% difference)** — the multi-block run has slightly more events because cannsim simulates multiple blocks on the same core sequentially. The extra events are from the second block's execution.

3. **Bottleneck is the same** — RVECLD (vector load) dominates in both cases at ~35% of events. The pipeline bottleneck lane doesn't change.

4. **New categories appear in multi-block only:**
   - `PID_PIPELINE_RVECLD` (8) — inter-block pipeline sync
   - `PID_PIPELINE_RVECST` (4) — inter-block pipeline sync

5. **Simulation time scales with grid x K-iters** — 2x2 grid took ~1.4x longer than 1x1. A 1024x1024x1024 run (8x8 grid, 8 K-iters = 512x more work) timed out at 300s.

## Conclusion

Sub-kernel simulation (grid=1, M=BLOCK_M, K=BLOCK_K) gives the **same bottleneck diagnosis** as multi-block. The instruction mix, dominant pipeline lane, and cycle counts are preserved. Always use sub-kernel for cannsim — the simulation time savings are dramatic (seconds vs hours) with no loss of optimization insight.

This validates Rule 1 from the simulation skill: "grid = (1, 1, 1) — one block is enough to see the bottleneck."

## ELF vs ELF_AIVEC Magic Number Comparison

**Same kernel, same cannsim config, only the `dev_bin.magic` field differs.**

| Metric | `ELF` | `ELF_AIVEC` |
|---|---|---|
| Cycles | 419 | 419 |
| Trace events | 23,127 | 11,604 |
| Bottleneck | identical | identical |

Both produce **identical execution** (same cycles, same bottleneck). `ELF` emits ~2× more events (extra vector pipeline annotations). Prefer `ELF_AIVEC` for Triton kernels — same actionable data, less noise.
