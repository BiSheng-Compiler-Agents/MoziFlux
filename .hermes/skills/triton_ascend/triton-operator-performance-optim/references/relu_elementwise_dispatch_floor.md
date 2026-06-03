# l1_19_ReLU Optimization Session — June 2026

## Kernel: standalone ReLU, fp16, (4096, 393216) benchmark shape

### cannsim Traces (sub-kernel: 1 tile, grid=1)

| Kernel | BLOCK_SIZE | wall_cycles | BOTTLENECK | RVECEX compute |
|--------|-----------|-------------|------------|----------------|
| Direct (3 runtime args) | 256 | 2910 | SCALARLDST 1713 cy | 22 cy (45:1 ratio) |
| Direct (constexpr pow2) | 256 | 2915 | SCALARLDST 1237 cy | 22 cy (LDP compensates) |
| Persistent | 4096 | 3327 | SCALARLDST 1699 cy | 125 cy (29:1 ratio) |

Per-element: persistent at BS=4096 → 0.81 cy/elem. Direct at BS=256 → 11.4 cy/elem.

### Fixed per-program SCALARLDST overhead breakdown

Every Triton program on Ascend pays this on startup (args-struct load):
- `DC_PRELOAD_XN_IMM` ~493 cy — icache + args prefetch
- `LDP_XI_XJ_XN` × 2 ~958 cy — pointer pair loads
- `LD_XD_XN_IMM` × 2–3 ~1460–2190 cy — scalar args (n_elements etc.)
- **Total ~3641 cy = ~1.46 µs fixed overhead per program**

At BS=256, N=1024: 4 programs × 3641 cy = 14,564 cy = 5.8 µs in SCALARLDST alone.
But hardware shows 42ms → the gap is CANN/Triton JIT dispatch stack overhead, not FFTS.

### Hardware benchmark results (v2 kernel)

| Shape | torch_ref | baseline | optimized | vs baseline |
|-------|-----------|----------|-----------|-------------|
| N=1024 | 1.4ms | 42ms | 54ms | 1.27× slower |
| N=65536 | 2.0ms | 39ms | 56ms | 1.43× slower |
| N=524K | 3.3ms | 40ms | 56ms | 1.39× slower |
| N=4M | 18.6ms | 46ms | 58ms | 1.24× slower |
| N=16M | 74ms | 93ms | 90ms | 0.97× (3% faster) |
| bench 1.6B | 4.5s | 0.9s* | 5.7s | 6.3× slower* |

*Baseline at bench shape was WRONG — it skipped 83% of elements (no loop, grid capped at 65535).
Corrected baseline for 1.6B elements is ~6× baseline above = ~5.4s.

### Why optimized can't beat baseline

1. `n_elements_pow2` extra arg: even as `tl.constexpr`, compiler compensates via LDP on SCALAR. No wall_cycle improvement.
2. fp32 upcast adds 2 `RV_VCVT_F2F` per sub-tile. Reduces WAIT_FLAG_VEC by ~6% per tile, but per-tile is masked by startup overhead at small N.
3. Persistent loop at small N: while-condition overhead without FFTS reduction.

At N=16M the 3% win comes from the fp32/propagate_nan improvements — the only shape where dispatch overhead is amortized enough for tile-level gains to show.

### Conclusion

Standalone elementwise Triton kernels on Ascend Ascend910_9589:
- Cannot compete with torch ACL at N < ~50M (dispatch dominates)
- Are near-optimal per tile at BS=4096 (0.81 cy/elem)
- The only path to beating torch: fuse with an adjacent operator
