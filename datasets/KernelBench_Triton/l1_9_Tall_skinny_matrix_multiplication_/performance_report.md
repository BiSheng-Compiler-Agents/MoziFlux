# Performance Report — Tall-Skinny Matrix Multiplication

## Test Configuration

| Parameter | Value |
|---|---|
| Kernel | `_matmul_kernel` (fp16, row-major) |
| Sub-kernel params | M=128, N=32, K=32 (1×BLOCK_K), grid=(1,1,1) |
| Autotune config | BLOCK_M=128, BLOCK_N=32, BLOCK_K=32, GROUP_M=8 |
| Simulator | CANN cannism (Ascend950 SOC) |
| Compile arch | Ascend910_9589 |
| Binary magic | RT_DEV_BINARY_MAGIC_ELF_AIVEC |

## Hardware Latency Estimate

Given `ref_period = 0.40 ns/cycle` on the target hardware:
- Baseline: 6686 cycles × 0.40 ns = **2.67 µs** per sub-kernel (1 tile, 1 K-iter)
- Optimized: 5921 cycles × 0.40 ns = **2.37 µs** per sub-kernel
- **Improvement: -11.4%** wall cycles

For the full benchmark shape (M=32768, N=32, K=32):
- Baseline: 6686 × (32768/128) × (32/32) = 6686 × 256 = **1,711,616 cycles ≈ 684.6 µs**
- Optimized: 5921 × (32768/128) × (32/32) = 5921 × 256 = **1,515,776 cycles ≈ 606.3 µs**

*Note: Full-shape latency is estimated assuming linear scaling from sub-kernel data. Actual hardware performance depends on FFTS dispatch, L2 cache effects, and multi-core scheduling.*

---

## Cannsim Trace Comparison

### Wall Cycles & Event Count

| Metric | Baseline | Optimized | Δ |
|---|---|---|---|
| wall_cycles | 6686 | 5921 | **-11.4%** |
| x_events | 2550 | 1602 | **-37.2%** |
| i_events | 112 | 88 | **-21.4%** |

### Pipeline Utilization (busy_cyc)

| Pipeline | Baseline | Optimized | Δ busy_cyc | Δ % |
|---|---|---|---|---|
| SCALARLDST | 3318 ← BOTTLENECK | 3246 ← BOTTLENECK | -72 | -2.2% |
| FLOWCTRL | 2595 | 1986 | **-609** | **-23.5%** |
| MTE3 | 2919 | 1942 | **-977** | **-33.5%** |
| PUSHQ | 2261 | 1069 | **-1192** | **-52.7%** |
| SCALAR | 1474 | 1461 | -13 | -0.9% |
| RVECST | 964 | 455 | **-509** | **-52.8%** |
| MTE2 | 934 | 926 | -8 | -0.9% |
| VEC | 836 | 845 | +9 | +1.1% |
| RVECLD | 774 | 437 | **-337** | **-43.5%** |
| RVECEX | 363 | 12 | **-351** | **-96.7%** |
| FIXP | 338 | 557 | +219 | +64.8% |
| MTE1 | 173 | 169 | -4 | -2.3% |
| CUBE | 170 | 166 | -4 | -2.4% |
| RVECSU | 119 | 117 | -2 | -1.7% |

### Top Instructions by Cycle Cost (CRITICAL items)

| Instruction | Pipe | Baseline (cnt / total) | Optimized (cnt / total) | Δ total |
|---|---|---|---|---|
| ST_XD_XN_IMM | SCALARLDST | 68 / 31205 | 74 / 44794 | +13589 |
| LD_XD_XN_IMM | SCALARLDST | 50 / 7701 | 46 / 8813 | +1112 |
| **RV_VSTI** | **RVECST** | **704 / 7901** | **320 / 3792** | **-4109** |
| **RV_VLDI** | **RVECLD** | **704 / 6592** | **320 / 2880** | **-3712** |
| **VF** | **PUSHQ** | **5 / 2089** | **2 / 1091** | **-998** |
| **SET_INTRA_BLOCKI** | **FLOWCTRL** | **7 / 2459** | **5 / 2027** | **-432** |
| WAIT_FLAG_VEC | MTE3 | 3 / 3721 | 2 / 3128 | -593 |
| DC_PRELOAD_XN_IMM | SCALAR | 4 / 2994 | 4 / 3036 | +42 |

### Instruction Count Improvements

| Instruction | Baseline | Optimized | Δ |
|---|---|---|---|
| RV_VSTI (vector store) | 704 | 320 | **-54.5%** |
| RV_VLDI (vector load) | 704 | 320 | **-54.5%** |
| VF (dispatch events) | 5 | 2 | **-60.0%** |
| SET_INTRA_BLOCKI (K-loop sync) | 7 | 5 | **-28.6%** |
| JUMPC | 34 | 26 | **-23.5%** |
| NOP | 10 | 4 | **-60.0%** |
| ST_XD_XN_IMM (scalar spill) | 68 | 74 | +8.8% |
| LD_XD_XN_IMM (scalar load) | 50 | 46 | -8.0% |

### Binary Size

| Metric | Baseline | Optimized | Δ |
|---|---|---|---|
| npubin size | 17928 bytes | 12936 bytes | **-27.8%** |

---

## Key Insights

1. **Dominant bottleneck shift**: Both baseline and optimized are bottlenecked by SCALARLDST (ST_XD_XN_IMM scalar spill). This is structural for Ascend matmul kernels with large BLOCK_M and remains the top bottleneck.

2. **Vector pipeline collapse**: The most dramatic improvement is in the vector pipeline. RVECEX busy_cyc dropped 96.7%, and RVECST/RVECLD dropped 54.5%. This directly results from `tl.dot(a, b, acc)` eliminating the 64 KB fp32 temporary tile.

3. **Instruction dispatch relief**: PUSHQ busy_cyc dropped 52.7% due to the `tl.range` loop eliminating advancing-pointer VF dispatch events.

4. **FLOWCTRL improvement**: SET_INTRA_BLOCKI reduced from 7 to 5 events (-28.6%) because the known-trip-count `tl.range` loop allows better compiler scheduling.

5. **SCALARLDST increased**: ST_XD_XN_IMM increased from 68→74 events. This appears to be a codegen artifact of the `tl.range` pattern — the compiler generates slightly more scalar spill instructions for the index-based pointer recomputation. This is a minor regression (~1.2% of wall time) that is offset by larger gains elsewhere.
