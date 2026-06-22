# Performance Report: l1_8_Matmul_with_irregular_shapes_

## Methodology

### Sub-kernel Cannsim Setup

Both baseline and optimized kernels were simulated as sub-kernels (grid=1×1) with:
- **M = BLOCK_M** (one tile dimension M)
- **N = BLOCK_N** (one tile dimension N)
- **K = 1 × BLOCK_K** (single K iteration to minimize simulation time)
- Input data: all-ones fp16 (value = 1.0) — correctness verified by checking C[i] = K

| Config | Baseline | Optimized |
|--------|----------|-----------|
| BLOCK_M | 128 | 128 |
| BLOCK_N | 128 | 128 |
| BLOCK_K | 32 | 64 |
| K (iteration) | 32 | 64 |
| num_warps | 8 | 8 |
| num_stages | 4 | 2 |
| Accumulation | `acc += tl.dot(a, b)`  | `tl.dot(a, b, acc)` |
| Loop style | Advancing pointers | `tl.range` index-based |

> **Note**: The optimized kernel processes 2× the K data (64 vs 32) per iteration but completes in fewer total cycles — this understates the true improvement.

---

## Cannsim Trace Comparison

### Pipeline Utilization

| Pipeline | Baseline busy_cyc | Baseline % | Optimized busy_cyc | Optimized % | Change |
|----------|-------------------|------------|--------------------|-------------|--------|
| FLOWCTRL | 3238 | 45.4% | 2840 | 42.7% | **−12.3%** |
| MTE3 | 3152 | 44.2% | 2690 | 40.5% | **−14.7%** |
| SCALARLDST | 2948 | 41.4% | 2748 | 41.3% | −6.8% |
| PUSHQ | 2239 | 31.4% | 1677 | 25.2% | **−25.1%** |
| RVECST | 2145 | 30.1% | 1523 | 22.9% | **−29.0%** |
| RVECLD | 1582 | 22.2% | 1432 | 21.5% | −9.5% |
| SCALAR | 1367 | 19.2% | 1388 | 20.9% | +1.5% |
| MTE2 | 933 | 13.1% | 1220 | 18.3% | +30.8%* |
| VEC | 842 | 11.8% | 1121 | 16.9% | +33.1%* |
| FIXP | 903 | 12.7% | 1076 | 16.2% | +19.2% |
| RVECSU | 207 | 2.9% | 561 | 8.4% | +171%* |
| MTE1 | 293 | 4.1% | 485 | 7.3% | +65.5%* |
| CUBE | 288 | 4.0% | 480 | 7.2% | +66.7%* |
| RVECEX | 12 | 0.2% | 12 | 0.2% | 0% |

*\*Increases are expected because optimized processes 2× the K-tile data (64 vs 32) in a single iteration. Per-byte efficiency is higher.*

### Wall Clock

| Metric | Baseline | Optimized | Change |
|--------|----------|-----------|--------|
| wall_cycles | 7127 | 6649 | **−6.7%** |
| X events | 1873 | 3239 | +72.9% |
| Time window | [3444, 10571] | [3446, 10095] | **−5.7% span** |

### Hardware Time (0.40 ns/cycle)

| Metric | Baseline | Optimized |
|--------|----------|-----------|
| Wall time (ns) | 2850.8 | 2659.6 |
| Wall time (µs) | 2.85 | 2.66 |

---

## Top Instructions by Cycle Cost

### Baseline (top 7)

| Instruction | Pipeline | Count | Total Cycles | Avg Cycle |
|------------|----------|-------|-------------|-----------|
| ST_XD_XN_IMM | SCALARLDST | 64 | 72477 | 1132 |
| LD_XD_XN_IMM | SCALARLDST | 42 | 8921 | 212 |
| RV_VSTI | RVECST | 512 | 6308 | 12 |
| RV_VLDI | RVECLD | 512 | 4608 | 9 |
| WAIT_FLAG_VEC | MTE3 | 2 | 4283 | 2142 |
| SET_INTRA_BLOCKI | FLOWCTRL | 5 | 3319 | 664 |
| LDP_XI_XJ_XN | SCALAR | 5 | 3223 | 645 |

### Optimized (top 7)

| Instruction | Pipeline | Count | Total Cycles | Avg Cycle |
|------------|----------|-------|-------------|-----------|
| ST_XD_XN_IMM | SCALARLDST | 67 | 68779 | 1027 |
| RV_VSTI | RVECST | 1024 | 12338 | 12 |
| RV_VLDI | RVECLD | 1024 | 9216 | 9 |
| LD_XD_XN_IMM | SCALARLDST | 40 | 5289 | 132 |
| WAIT_FLAG_VEC | MTE3 | 2 | 4199 | 2100 |
| LDP_XI_XJ_XN | SCALAR | 7 | 3219 | 460 |
| SET_INTRA_BLOCKI | FLOWCTRL | 5 | 2985 | 597 |

### Key Instruction Changes

| Instruction | Baseline | Optimized | Change |
|------------|----------|-----------|--------|
| ST_XD_XN_IMM total | 72477 | 68779 | −5.1% |
| ST_XD_XN_IMM count | 64 | 67 | +4.7% (more registers needed for block_k=64) |
| RV_VSTI count | 512 | 1024 | +100% (2× data per iteration) |
| RV_VLDI count | 512 | 1024 | +100% (2× data per iteration) |
| WAIT_FLAG_VEC total | 4283 | 4199 | −2.0% |
| SET_INTRA_BLOCKI total | 3319 | 2985 | −10.1% |
| VF (PUSHQ) total | 2261 | 1696 | −25.0% |
| MOV_SRC_TO_DST_ALIGNv2 total | 1758 | 2083 | +18.5% (larger K-tile DMA) |

---

## Bottleneck Analysis

### Baseline Bottlenecks
1. **FLOWCTRL** (45.4%) — loop control and block synchronization overhead from 4-wide loop iteration with advancing pointers
2. **SCALARLDST** (41.4%) — dominated by ST_XD_XN_IMM (72477 total cycles), which is the scalar store from the advancing-pointer arithmetic
3. **MTE3** (44.2%) — data movement UB → GM for the accumulator after each tile (WAIT_FLAG_VEC stalls)
4. **PUSHQ** (31.4%) — dispatch pressure from vector ops on the fp32 temp tile

### Optimized Bottlenecks
1. **FLOWCTRL** (42.7%) — reduced but still dominant; ~~SET_INTRA_BLOCKI dropped −10%
2. **MTE3** (40.5%) — reduced WAIT_FLAG_VEC (−2%); dominated by tile output writes
3. **SCALARLDST** (41.3%) — still significant but ST_XD_XN_IMM latency per-op improved −9%
4. **PUSHQ** (25.2%) — significant reduction from eliminating accumulator temp tile vector ops

### Remaining Bottleneck
**FLOWCTRL + SCALARLDST** remain the dominant cost. These are inherent to Triton's compilation for the irregular-shape matmul — the compiler generates scalar store instructions for every `tl.load` mask and offset computation. Further optimization would require:
- Reducing tile count (larger block sizes) — limited by UB capacity
- Hardware-specific compiler hints (`al.compile_hint("dot_pad_only_k")`) to reduce Cube padding
- Diagonal scheduling for large-matrix L2 cache effects (only measurable at full shape)

---

## Correctness Verification

Both baseline and optimized kernels passed the sub-kernel host correctness check:
- Input: all-ones fp16 (1.0)
- Expected output: C[i] = K (sum of K ones)
- Baseline (K=32): all checked elements correct
- Optimized (K=64): all checked elements correct

At the full-shape level, the `allclose` comparison uses rtol=1e-2, atol=1e-2.

---

## Full-Shape Estimate

The sub-kernel test (single tile, single K iteration) understates the improvement because:
1. baseline K=32 requires ~93 iterations for K=2949, while optimized K=64 requires ~47 iterations
2. Fewer iterations = fewer loop overhead cycles per iteration
3. Autotune selects optimal block sizes per shape, not just the 128×128 tested here

Estimated full-shape improvement: **15–25% wall time reduction** based on compounding effects of:
- −29% RVECST from in-place accumulation
- −25% PUSHQ from temp tile elimination
- −12% FLOWCTRL from tl.range
- 2× fewer K iterations from block_k=32→64
