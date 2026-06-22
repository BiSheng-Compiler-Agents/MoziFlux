# Performance Report — 3D Tensor Matrix Multiplication (V2)

## Test Environment

- **Simulator**: CANN cannsim (Ascend950)
- **Target**: Ascend950 (cannsim), compiled with `Ascend910_9589` ISA
- **Sub-kernel scale**: grid=(1,1), 1 tile per kernel, 2 K-loop iterations
- **Data type**: fp16 input, fp32 accumulator, fp16 output

### Baseline Configuration
- BLOCK_M=128, BLOCK_N=128, BLOCK_K=32
- `while k_iter < K` loop, `a.to(tl.float32)` before tl.dot
- grid=(1,1), M=128, N=128, K=64 (2 iterations of BLOCK_K=32)

### Optimized V2 Configuration
- BLOCK_M=128, BLOCK_N=128, BLOCK_K=64, GROUP_M=8
- `tl.range(0, num_k_iters)` loop, `tl.dot(a, b, acc)` in-place
- grid=(1,1), B=1, M=128, N=128, K=128 (2 iterations of BLOCK_K=64)

---

## Cannsim Trace Summary

### Baseline Trace
```
wall_cycles: 54,666  |  x_events: 62,107  |  i_events: 13,506

Pipeline Utilization:
  PUSHQ    47,710 busy_cyc  ← BOTTLENECK (87.3% of wall)
  SCALAR   27,770 busy_cyc
  RVECST   27,426 busy_cyc
  RVECLD   27,373 busy_cyc
  FLOWCTRL 15,841 busy_cyc
  RVECEX   10,137 busy_cyc
  MTE3      8,633 busy_cyc
  MTE2      4,163 busy_cyc
  VEC       4,011 busy_cyc
  SCALARLDST  3,586 busy_cyc
  FIXP      2,830 busy_cyc
  CUBE      2,273 busy_cyc  (4.2% of wall)

Top Instructions:
  VF@PUSHQ         2,586 events  avg  69 cyc  total 178,194 cyc  ★ CRITICAL
  ST_XD_XN_IMM      213 events  avg 478 cyc  total 101,914 cyc
  RV_VSTI@RVECST  4,361 events  avg  10 cyc  total  44,766 cyc
  RV_VLDI@RVECLD  3,851 events  avg   9 cyc  total  35,432 cyc
  SHL             5,493 events  avg   4 cyc  total  21,972 cyc
  ADD_IMM         5,400 events  avg   4 cyc  total  21,600 cyc
  ZEROEXT         5,378 events  avg   4 cyc  total  21,512 cyc
  SET_INTRA_BLOCKI  11 events  avg 1,328 cyc  total  14,604 cyc
  WAIT_FLAG_VEC      2 events  avg 6,653 cyc  total  13,306 cyc
```

### Optimized V2 Trace
```
wall_cycles: 8,689  |  x_events: 6,206  |  i_events: 265

Pipeline Utilization:
  FLOWCTRL  3,880 busy_cyc  ← BOTTLENECK (44.7% of wall)
  SCALARLDST  3,675 busy_cyc
  MTE3      3,617 busy_cyc
  PUSHQ     3,353 busy_cyc
  MTE2      3,093 busy_cyc
  RVECST    3,046 busy_cyc
  SCALAR    2,962 busy_cyc
  RVECLD    2,864 busy_cyc
  VEC       2,258 busy_cyc
  RVECSU    1,122 busy_cyc
  FIXP      1,056 busy_cyc
  CUBE        956 busy_cyc  (10.5% of wall)
  MTE1        642 busy_cyc
  RVECEX       24 busy_cyc

Top Instructions:
  ST_XD_XN_IMM       79 events  avg 468 cyc  total 36,968 cyc  ★ CRITICAL
  RV_VSTI@RVECST  2,048 events  avg  12 cyc  total 24,676 cyc
  RV_VLDI@RVECLD  2,048 events  avg   9 cyc  total 18,432 cyc
  WAIT_FLAG_VEC       4 events  avg 1,104 cyc  total  4,415 cyc
  SET_INTRA_BLOCKI    8 events  avg   518 cyc  total  4,148 cyc
  VF@PUSHQ            4 events  avg   841 cyc  total  3,364 cyc
  WAIT_FLAG_VEC       6 events  avg   570 cyc  total  3,419 cyc
```

---

## Key Performance Comparisons

### Wall Cycles

| Metric | Baseline | Optimized V2 | Improvement |
|--------|----------|-------------|-------------|
| wall_cycles | 54,666 | 8,689 | **6.3×** |
| x_events | 62,107 | 6,206 | **10.0×** |
| i_events | 13,506 | 265 | **51.0×** |

### Pipeline Breakdown

| Pipeline | Baseline (busy_cyc) | Optimized V2 (busy_cyc) | Change |
|----------|---------------------|------------------------|--------|
| PUSHQ (dispatch) | 47,710 | 3,353 | **−93.0%** |
| SCALAR | 27,770 | 2,962 | **−89.3%** |
| RVECEX | 10,137 | 24 | **−99.8%** |
| FLOWCTRL | 15,841 | 3,880 | **−75.5%** |
| MTE3 | 8,633 | 3,617 | **−58.1%** |
| CUBE | 2,273 | 956 | **−57.9%** |
| MTE2 | 4,163 | 3,093 | **−25.7%** |

### Instruction-Level Comparison

| Instruction | Baseline (cnt/avg/total) | Optimized V2 (cnt/avg/total) | Change |
|-------------|-------------------------|-----------------------------|--------|
| VF (PUSHQ) | 2,586 / 69 / 178,194 | 4 / 841 / 3,364 | **−99.8% cnt** |
| RV_VSTI | 4,361 / 10 / 44,766 | 2,048 / 12 / 24,676 | **−53.0% cnt** |
| RV_VLDI | 3,851 / 9 / 35,432 | 2,048 / 9 / 18,432 | **−46.8% cnt** |
| ST_XD_XN_IMM | 213 / 478 / 101,914 | 79 / 468 / 36,968 | **−62.9% cnt** |
| SET_INTRA_BLOCKI | 11 / 1,328 / 14,604 | 8 / 518 / 4,148 | **−27.3% cnt, −61.0% avg** |
| WAIT_FLAG_VEC | 2 / 6,653 / 13,306 | 4 / 1,104 / 4,415 | **−83.4% avg** |
| SHL | 5,493 / 4 / 21,972 | 0 | **−100%** |
| ADD_IMM | 5,400 / 4 / 21,600 | 0 | **−100%** |
| ZEROEXT | 5,378 / 4 / 21,512 | 0 | **−100%** |
| JUMPC | 5,180 | 50 | **−99.0%** |

---

## V1 vs V2 Comparison (in-place dot + tl.range)

The V1 optimized kernel (mask hoisting, multibuffer, but `acc += tl.dot(a,b)` and `while` loop) achieved 14,141 cycles. V2 adds two critical optimizations:

| Metric | V1 (14,141 cyc) | V2 (8,689 cyc) | Δ |
|--------|-----------------|----------------|-----|
| wall_cycles | 14,141 | 8,689 | **−38.5%** |
| RVECEX busy_cyc | 1,219 | 24 | **−98.0%** |
| RVECLD ops | 3,328 | 2,048 | **−38.5%** |
| RVECST ops | 3,072 | 2,048 | **−33.3%** |
| FLOWCTRL busy_cyc | 9,105 | 3,880 | **−57.4%** |
| SET_INTRA_BLOCKI avg | 913 cyc | 518 cyc | **−43.3%** |
| VF events | 8 | 4 | **−50.0%** |

### What Changed
1. **`acc += tl.dot(a, b)` → `acc = tl.dot(a, b, acc)`**: Eliminated 64 KB temporary tile, saving 98% of RVECEX and 33-38% of vector load/store traffic.
2. **`while k_iter < K` → `for k_idx in tl.range(0, num_k_iters)`**: Halved FLOWCTRL busy_cyc, eliminated 99% of JUMPC instructions.
3. **Removed `al.multibuffer`**: In-place dot makes double buffering redundant and eliminates the UB overflow risk on real hardware.

---

## Bottleneck Analysis

### Baseline Bottleneck: PUSHQ (Instruction Dispatch Pressure)
PUSHQ at 47,710 busy_cyc (87.3% of wall) — driven by 2,586 VF instructions from:
- Per-iteration mask computation (5,493+ SHL, 5,400+ ADD_IMM, 5,378+ ZEROEXT)
- `a.to(tl.float32)` conversion generating numerous casting instructions
- Extensive scalar register spill from pointer arithmetic (213 ST_XD_XN_IMM)
- 5,180 JUMPC from while-loop control flow

### Optimized V2 Bottleneck: FLOWCTRL (SET_INTRA_BLOCKI)
FLOWCTRL at 3,880 busy_cyc (44.7% of wall) — dominated by SET_INTRA_BLOCKI (8 events avg 518 cyc). This is an unavoidable cost of Triton's structured control flow on Ascend. The per-event duration dropped 61% from baseline due to the reduced loop body after mask hoisting and in-place accumulation.

The bottleneck shift (PUSHQ → FLOWCTRL) indicates that the main sources of per-iteration overhead have been eliminated. Remaining gain opportunities require reducing the structural SET_INTRA_BLOCKI cost or amortizing FFTS dispatch via persistent/work-stealing grids.

---

## Performance Ratio (Per-Tile Normalized)

| Metric | Baseline | Optimized V2 | Improvement |
|--------|----------|-------------|-------------|
| wall_cycles | 54,666 | 8,689 | 6.3× |
| Inner product size | 128×64 (8,192) | 128×128 (16,384) | 2.0× |
| Cycles per element | 54,666 / 8,192 = 6.67 | 8,689 / 16,384 = 0.53 | **12.6×** |
| CUBE utilization | 4.2% of wall | 10.5% of wall | +6.3pp |

---

## Hardware Latency Estimate (Ascend950)

Using `ref_period = 0.40 ns/cycle`:

| Metric | Baseline | Optimized V2 | Improvement |
|--------|----------|-------------|-------------|
| wall_cycles | 54,666 | 8,689 | 6.3× |
| Estimated wall time (ns) | 21,866 ns | 3,476 ns | 6.3× |
| Estimated wall time (µs) | 21.9 µs | 3.5 µs | 6.3× |

> **Note**: These are sub-kernel estimates (1 tile, 2 K-iterations) at sub-kernel scale. Full-shape performance depends on grid size, FFTS dispatch cost, and L2 cache effects not captured at grid=1.

---

## Summary

| Category | Result |
|----------|--------|
| **Wall-cycle speedup** | **6.3×** (54,666 → 8,689) |
| **Per-element throughput** | **12.6×** (6.67 → 0.53 cyc/elem) |
| **VF instruction reduction** | **99.8%** (2,586 → 4) |
| **SCALAR op reduction** | **89.3%** (27,770 → 2,962 busy_cyc) |
| **CUBE utilization** | 4.2% → 10.5% of wall |
| **Dominant bottleneck** | PUSHQ (87.3%) → FLOWCTRL (44.7%) |
| **V1 → V2 improvement** | −38.5% (14,141 → 8,689) from in-place dot + tl.range |

The V2 transformation shifts the kernel from instruction-dispatch-bound (87.3% PUSHQ) to control-flow-bound (44.7% FLOWCTRL). The two critical V2 additions — `tl.dot(a, b, acc)` in-place accumulation and `tl.range` loop — together produce a **38.5% reduction** over V1 and a **6.3× improvement** over the baseline.
