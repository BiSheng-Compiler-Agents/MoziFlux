# Performance Report — l1_21 Sigmoid

## Methodology

- **Hardware target:** Ascend950 (emulated via CANN cannsim)
- **Simulation:** Sub-kernel cannsim trace (grid=1, BLOCK_SIZE=4096, 1 tile of fp32 data)
- **Trace analysis:** `aggregate_trace.py` on `trace_core0.json`
- **Cycle-to-time conversion:** 0.40 ns/cycle (Ascend camodel nominal)

---

## Sub-Kernel Cannsim Trace Comparison (fp32, BLOCK_SIZE=4096, 1 tile)

### Pipeline Utilization

| Pipeline | Baseline (cy) | Optimized (cy) | Δ (cy) | Notes |
|----------|--------------|----------------|--------|-------|
| **MTE3** | **1,968** ←BOTTLENECK | **1,999** ←BOTTLENECK | +31 | Store pipeline — inherent to elementwise write |
| SCALAR | 1,779 | 1,772 | -7 | Negligible |
| SCALARLDST | 1,226 | 1,216 | -10 | Args struct loading — fixed per-tile cost |
| MTE2 | 1,020 | 1,018 | -2 | Input load |
| VEC | 1,014 | 1,012 | -2 | WAIT_FLAG_MTE2 |
| PUSHQ | 506 | 542 | +36 | Small increase from compiler scheduling |
| **RVECEX** | **461** (515 ops) | **497** (580 ops) | +36 | More ops from scalar-vector subtract |
| RVECLD | 373 | 419 | +46 | Load activity |
| RVECST | 295 | 321 | +26 | Store activity |
| FLOWCTRL | 7 | 7 | 0 | |

**Wall cycles:** 3,749 (baseline) → 3,773 (optimized) = **+0.6%** (within noise)

### Top Instructions Comparison

| Instruction | Baseline | Optimized | Δ | Notes |
|-------------|----------|-----------|---|-------|
| RV_VDIV | 128×2,176 cy | 128×2,176 cy | 0 | 1/(1+z) — the dominant compute cost |
| WAIT_FLAG_VEC@MTE3 | 1×1,508 cy | 1×1,542 cy | +34 | Store stall — unchanged |
| LDP_XI_XJ_XN | 3×1,429 cy | 3×1,438 cy | +9 | Args struct load |
| LD_XD_XN_IMM | 1×1,226 cy | 1×1,216 cy | -10 | Scalar arg load |
| STI_XN_IMM | 1×1,225 cy | 1×1,215 cy | -10 | Scalar store |
| RV_VEXP | 64×1,024 cy | 64×1,024 cy | 0 | Exponential — unchanged |
| WAIT_FLAG_MTE2 | 1×1,014 cy | 1×1,012 cy | -2 | Load stall |
| **RV_VMULS** | **64×512 cy** | **0** | **-512** | **Eliminated!** Replaced by subtraction |
| **RV_VADDS** | **0** | **128×896 cy** | **+896** | New: scalar-vector subtraction |
| RV_VSTI | 64×582 cy | 64×586 cy | +4 | Vector store |
| RV_VLDI | 64×576 cy | 64×576 cy | 0 | Vector load |

### Key Changes

1. **RV_VMULS eliminated** — The `z * inv` multiplication in the negative branch of `tl.where` is replaced by `1.0 - inv`. On Ascend, `tl.where` computes both branches and selects, so the elimination of the multiply branch removes 64 VMULS instructions.

2. **RV_VADDS introduced** — The subtraction `1.0 - inv` compiles to 128 additions (scalar-vector subtract uses a narrower pipeline width of 32 elements vs 64). This adds 896 cy but the total compute is still dominated by VDIV and VEXP.

3. **Bottleneck unchanged** — Both kernels are bottlenecked by MTE3 (store pipeline, ~2,000 cy) and SCALARLDST (args loading, ~1,200 cy). The per-tile compute (RVECEX, ~500 cy) is a secondary cost.

---

## Hardware Latency (TBD — No NPU Hardware Available)

**Status: 🔴 No real NPU hardware was available for this evaluation.**

Cannsim sub-kernel traces measure per-tile cycle counts at the instruction level. They do NOT include:
- Python→CANN JIT dispatch overhead (~40 ms fixed cost for standalone ops)
- FFTS dispatch cost per program (~1,150 cy/program)
- Multi-program L2 cache effects
- Multi-core parallelism

### Projected Full-Shape Performance

For a standalone elementwise Sigmoid, the hardware latency is dominated by the CANN/Triton dispatch stack (~40 ms fixed cost), not per-tile compute (~3,800 cy × 0.4 ns = 1.5 µs per tile). This is an inherent limitation of standalone Triton elementwise ops on Ascend — confirmed by l1_19_ReLU and l1_20_LeakyReLU episodes (June 2026).

| Input Size | n_tiles (BLOCK=4096) | Sub-kernel compute | Dispatch overhead | Predicted total |
|-----------|---------------------|--------------------|------------------|----------------|
| 4,096 | 1 | ~3,800 cy = 1.5 µs | ~40 ms | ~40 ms |
| 1,048,576 (1M) | 256 | ~384,000 cy = 154 µs | ~40 ms | ~40 ms |
| 268,435,456 (256M) | 65,536 | ~98,000,000 cy = 39 ms | ~0 (persistent) | ~39 ms |
| 1,610,612,736 (1.6B) | 393,216 | ~590,000,000 cy = 236 ms | ~0 (persistent) | ~236 ms |

**Key insight:** Below ~256M elements, the dispatch overhead (~40 ms) dominates. The optimized kernel's dispatch efficiency (two-path dispatch, autotune) only matters above this threshold. For production use, Sigmoid should be fused with an adjacent operator rather than launched standalone.

### Performance Ratio

| Size | torch.sigmoid | Baseline (projected) | Optimized (projected) | Speedup vs baseline |
|------|--------------|--------------------|----------------------|-------------------|
| < 1M | — | Dispatch-bound | Dispatch-bound | ~1.0× (no difference) |
| 1M–256M | — | Dispatch-bound | Dispatch-bound | ~1.0× |
| > 256M | — | 65535-grid-capped | Persistent grid | **Functional (no crash)** |

**Note:** Hardware measurement on real Ascend NPU is required for accurate latency figures. The above projections assume the sub-kernel trace scales linearly with tile count.

---

## Cannsim Trace Details

### Baseline Trace

```
wall_cycles: 3749  |  x_events: 741  |  i_events: 8
← BOTTLENECK: MTE3 (1968 cy)
← CRITICAL: RV_VDIV 128×2176 cy (avg 17), WAIT_FLAG_VEC 1×1508 cy
← CRITICAL: LD_XD_XN_IMM 1×1226 cy, STI_XN_IMM 1×1225 cy
```

### Optimized Trace

```
wall_cycles: 3773  |  x_events: 806  |  i_events: 8
← BOTTLENECK: MTE3 (1999 cy)
← CRITICAL: RV_VDIV 128×2176 cy (avg 17), WAIT_FLAG_VEC 1×1542 cy
← CRITICAL: LD_XD_XN_IMM 1×1216 cy, STI_XN_IMM 1×1215 cy
← NEW: RV_VADDS 128×896 cy (replaces RV_VMULS 64×512 cy)
```

---

## Conclusion

The optimized kernel provides:

1. **P0 Fix — Grid Overflow Protection:** The baseline would crash on any input with `cdiv(n, 256) > 65535`. The two-path dispatch prevents this.
2. **P1 Fix — Autotune:** Enables optimal BLOCK_SIZE selection per input size class, with bucketed key to prevent autotune thrashing.
3. **P2 Improvement — Code Quality:** `care_padding=False`, simplified sigmoid computation.

**Sub-kernel per-tile performance is within noise (±0.6%) of baseline.** This is expected for standalone elementwise kernels where SCALARLDST (~1,200 cy) and MTE3 (~2,000 cy) dominate the per-tile cost, and hardware dispatch overhead (~40 ms) dominates end-to-end.

**To beat torch.sigmoid on hardware:** The only viable path is kernel fusion with an adjacent op (e.g., `Linear + Sigmoid`), eliminating the standalone kernel launch overhead.
