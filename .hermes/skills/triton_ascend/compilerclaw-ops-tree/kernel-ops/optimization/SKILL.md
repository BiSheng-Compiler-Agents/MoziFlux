---
name: optimization
description: Optimize Ascend NPU-native Triton operator performance. Use when diagnosing bottlenecks from cannsim traces and applying optimization patterns. Covers UB overflow detection, Cube utilization improvement, tiling strategy design, and pattern application.
tags: [triton, ascend, optimization, performance]
---

# Triton Kernel Performance Optimization [LEAF NODE]

Optimize Ascend NPU-native Triton operator performance. Use when diagnosing bottlenecks
from cannsim traces and applying optimization patterns. Covers UB overflow detection,
Cube utilization improvement, tiling strategy design, and pattern application.

## Bottom Lines (Not to be Broken)

1. **Precision**: After optimization, rtol=1e-3, atol=1e-3 must align with PyTorch-NPU. Roll back if not met.
2. **Generalization**: Support all original input shapes and dtypes; do not hardcode specific sizes.

**Performance Ratio Definition**: `Ratio = torch_npu time / Triton time`. Ratio > 1.0 means Triton is faster.
**Priority**: Correctness > Generalization > Performance.

### Generalization Rules (mandatory — violations are P0)

- **Never read shapes from benchmark files, perf reports, or other operator files** when writing the optimized kernel. The only source of truth for what shapes must be supported is the original kernel's function signature and the operator's mathematical definition.
- **Multiple kernel variants dispatched by the host are fine** (e.g. a fast no-mask path for power-of-2 C alongside a general masked path for all other C). What is not allowed is a variant that only handles specific shapes and leaves other shapes broken or unhandled.
- **Every dispatch path must be correct and tested** — if you write a C=16 fast path and a generic path, both must have unit tests. Never ship an untested code path.
- **Never add new runtime guards that the baseline did not have** (e.g. `if out_channels > 256: raise`, `if sum_dim != 1: raise`). If the baseline kernel accepted a parameter freely, the optimized kernel must too.
- **Cover all parameter combinations in unit tests**: small/large/non-power-of-2 sizes for every free dimension in the signature. Never test only the benchmark shape.

---

## Optimization Workflow

### Phase 0: Algorithm Review

Review the algorithm itself before optimization. An inefficient algorithm has inherent
limitations no matter how much you optimize.

### Phase 1: Retrieve Past Episodes

Before writing any code, query episodes for relevant patterns:

```python
episode_retrieve(query="<kernel_type> <bottleneck>", target="ascend950", limit=5)
# Examples:
episode_retrieve(query="matmul cube utilization dot pad static range", target="ascend950")
episode_retrieve(query="softmax wide rows MTE online reduction", target="ascend950")
episode_retrieve(query="norm scalar overhead two-pass single-pass", target="ascend950")
episode_retrieve(query="elementwise FFTS dispatch persistent grid", target="ascend950")
```

### Phase 2: Hierarchical Evaluation

1. **Quick Screening**: If real NPU hardware available, measure end-to-end with `time.time()`.
2. **Precise Diagnosis**: Use `cannsim_remote_run(gen_report=True)` with a **sub-kernel host**
   (grid=1, M=BLOCK_M, K=2×BLOCK_K — see simulation/SKILL.md Rule 1). This makes each
   cannsim run take seconds instead of minutes/hours while giving identical bottleneck diagnosis.
   Then run:
   ```bash
   python <tree_root>/kernel-ops/simulation/scripts/aggregate_trace.py \
       /path/to/report/trace_core0.json
   ```

### Phase 3: Bottleneck Optimization

| Bottleneck | Optimization Focus |
|------------|---------------------|
| Memory-Bound (MTE2 > 50%) | Vectorized memory access, UB cache reuse, operator fusion |
| Compute-Bound (aiv_vec > 50%) | Cube adaptation, block size tuning |
| Scalar-Bound (aiv_scalar > 80%) | Single-pass reduction, eliminate div/mod, tl.zeros accumulators |
| Latency-Bound | Increase parallelism, reduce synchronization |

**Four Fundamental Moves** (in order):
1. Block/Grid Size Tuning
2. Contiguous Memory Access
3. UB Reuse (single-pass)
4. Compile-time Constants

---

## Hardware Constraints Quick Reference

| Resource | Value | Notes |
|---|---|---|
| Unified Buffer (UB) | 192 KB | Per AI Core. All active tensors must fit simultaneously. |
| L1 Buffer | 1 MB | Cube core only. |
| 32-byte alignment | Required | All load/store addresses and buffer starts. |
| Cube granularity | 16×16 minimum | BLOCK_M, BLOCK_N, BLOCK_K must all be multiples of 16. |
| UB safety factor | ×0.65–0.8 | Compiler adds ~15–35% overhead beyond estimates. |
| Max grid (1D) | `num_aicore` or `num_vectorcore` | Exceeding adds host scheduling overhead. |

**UB estimation:**
```python
def estimate_ub_bytes(BLOCK_M, BLOCK_N, BLOCK_K, dtype_size=2):
    a_tile = BLOCK_M * BLOCK_K * dtype_size
    b_tile = BLOCK_K * BLOCK_N * dtype_size
    acc    = BLOCK_M * BLOCK_N * 4            # FP32 accumulator
    masks  = BLOCK_M * BLOCK_N                # mask array
    return a_tile + b_tile + acc + masks

AVAILABLE_UB = int(192 * 1024 * 0.65)  # ~128 KB usable
# For 2D tiling: also add offset arrays: 2 * ROWS * COLS * 4 (int32)
```

`next_power_of_2` trap: `triton.next_power_of_2(48) = 64`, which may exceed UB.
Use floor rounding: `1 << (n.bit_length() - 1)`.

---

## Core Optimization Rules

### Rule 1: Block Sizing
- Vector ops: BLOCK_SIZE 1024–2048 for FP16 to fit UB with headroom
- Matrix ops: BLOCK_M/N/K multiples of 16. Safe starting point: M=128, N=256, K=64–256
- Use `@triton.autotune` to sweep block sizes
- Grid: 1D preferred, equal to physical core count

### Rule 2: Memory Access Contiguity (highest-impact rule)

Contiguous access enables the compiler to merge many small transactions into large-block DMA.
Non-contiguous patterns force many tiny MTE instructions (can cause 86× more DMA ops).

```python
# Good: contiguous, compiler can merge into large DMA
offsets = block_start + tl.arange(0, BLOCK_SIZE)

# Bad: non-contiguous
offsets = block_start + tl.arange(0, BLOCK_SIZE) * stride

# Bad: 2D broadcast creates non-contiguous pattern
off = row_off[:, None] + col_off[None, :]   # prevents DMA merging

# Good: host-side expand+contiguous eliminates broadcast stride
cos_flat = cos.expand(x_shape).contiguous().reshape(total_rows, D)
# Now row * D + col is contiguous — DMA engine runs at full speed
```

### Rule 3: Single Pass Over Multi-Pass

Loading the same data multiple times multiplies scalar overhead. When the whole row fits in UB:

```python
# Bad: 3 loads of x, huge scalar overhead
# Good: load once, all computation in UB
x = tl.load(x_ptr + tot_off, mask=mask, other=0.0).to(tl.float32)
sum_x  = tl.sum(x, 1)
mean   = sum_x / D
var    = tl.sum(x * x, 1) / D - (mean ** 2)
x_norm = (x - mean[:, None]) * tl.rsqrt(var + eps)[:, None]
tl.store(y_ptr + tot_off, x_norm * w + b, mask=mask)
```

NBLOCK can be as large as 8192 when D ≤ UB capacity / num_buffers.

**`care_padding=False`** — skip padding check for ~5–10% free speedup:
```python
x = tl.load(ptr + offsets, mask=mask, other=0.0, care_padding=False)
```
Safe when padding values do not affect downstream computation.

### Rule 4: Operator Fusion

Eliminating intermediate GM round-trips transforms memory-bound → compute-bound:
```python
# Before: 2 GM round-trips
# After: 1 GM round-trip (load x → relu → softmax → store w)
x = tl.load(x_ptr + offsets, mask=mask)
w = tl.softmax(tl.where(x > 0, x, 0.0).to(tl.float32))
tl.store(y_ptr + offsets, w.to(tl.float16), mask=mask)
```

### Rule 5: Precision Rules

```python
# All reductions: upcast to FP32 first
x_fp32 = x.to(tl.float32)
mean = tl.sum(x_fp32, axis=-1) / D

# Matrix multiply: FP16 load, FP32 accumulate, FP16 store
acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
acc += tl.dot(a_fp16, b_fp16)
tl.store(c_ptr + ..., acc.to(tl.float16))
```

### Rule 6: Compile-Time Constants

```python
def kernel(x_ptr, BLOCK_SIZE: tl.constexpr):
    for i in tl.static_range(0, BLOCK_SIZE):  # static unroll
        ...
mask = offsets < n_elements
x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
```

### Rule 7: Intra-Core Tiling When UB Is Tight

```python
for sub_start in range(0, BLOCK_SIZE, SUB_BLOCK_SIZE):
    offsets = start + sub_start + tl.arange(0, SUB_BLOCK_SIZE)
    mask = offsets < n_elements
    x_chunk = tl.load(x_ptr + offsets, mask=mask)
    tl.store(y_ptr + offsets, process(x_chunk), mask=mask)
```

### Rule 8: Two-Path Dispatch for Elementwise Kernels (MANDATORY)

**The trap:** "use a persistent grid" is a common elementwise recommendation
(see episodes 12, 43, 46, 49). It is **only a win when the natural tile count
exceeds the FFTS grid cap (65535)**. Below that threshold it is a regression.

| n_tiles (BLOCK=4096) | n_elements | Best path | Why |
|---|---|---|---|
| ≤ 65,535 | ≤ 268,431,360 (~256M) | **direct** (one program per tile) | Fastest dispatch; JUMPC overhead has no upside |
| > 65,535 | > 256M | **persistent** (work-stealing while loop) | Direct would crash (coredim > UINT16_MAX) or saturate FFTS |

**Hardware-validated (l1_19_ReLU, l1_5 matrix-scalar, June 2026):**
- When `n_tiles ≤ 65535`, the while-loop **adds JUMPC overhead with zero
  FFTS reduction** — measured **1.35–1.43× SLOWER** than direct dispatch.
- When `n_tiles > 65535`, persistent gives **~1.42–4× speedup** from
  amortising the ~1,150 cy/program FFTS dispatch cost.

**Implementation:**

```python
_MAX_PROGRAMS = 65535  # Ascend FFTS grid cap

class ModelNew(nn.Module):
    def forward(self, x):
        n_tiles = triton.cdiv(x.numel(), BLOCK_SIZE)
        if n_tiles > _MAX_PROGRAMS:
            # Persistent path — cap grid, each program strides
            n_programs = _MAX_PROGRAMS
            _kernel_persistent[(n_programs,)](
                ..., n_programs=n_programs, BLOCK_SIZE=BLOCK_SIZE)
        else:
            # Direct path — one program per tile, no while loop
            _kernel_direct[(n_tiles,)](
                ..., BLOCK_SIZE=BLOCK_SIZE)
```

**Routing threshold — common mistake:** do NOT use `n > SOME_NUMBER`. Use
`cdiv(n_elements, BLOCK_SIZE) > _MAX_PROGRAMS`. The threshold depends on
`BLOCK_SIZE`:
- `BLOCK=4096` → threshold at n = 268,431,360 (~256M)
- `BLOCK=256`  → threshold at n = 16,776,960 (~16M)

If you use `@triton.autotune` with multiple `BLOCK_SIZE` configs, the routing
threshold MUST use the **smallest** BLOCK in the configs (per episode 46
finding — using the largest leads to a runtime `coreDim > UINT16_MAX` crash
on the first shape that triggers the persistent path). Pattern:
`threshold = cdiv(n, min_BLOCK) > MAX_PROGRAMS`.

**Episode references for this pattern:**
- Episode 12 — first formalization of the persistent loop pattern + threshold
- Episode 43 — confirmed for matrix-scalar-multiplication, sub-kernel trace
  shows persistent is structurally identical to direct at grid=1 (FFTS benefit
  invisible)
- Episode 46 — ReLU: revealed that persistent is harmful at small N,
  established the routing threshold rule
- Episode 49 — matrix-scalar: prior opt was unconditionally persistent and
  was a regression; fixed with two-path dispatch

**Sub-kernel trace is identical for direct vs persistent** at grid=1 — the
benefit is purely at full-shape FFTS dispatch level. Do not conclude the
optimization failed from a per-tile cycle delta; verify the kernel works
correctly and let hardware measurements confirm the dispatch savings.

---

## Ascend-Specific Compiler Hints

```python
import triton.language.extra.cann.extension as al

# dot_pad_only_k: pad only K dimension, reducing UB usage 30-50%
al.compile_hint(acc, "dot_pad_only_k")

# Double buffering (overlaps DMA with compute)
a = al.multibuffer(a, size=2)  # only size=2 supported

# Type conversion with overflow control
y = al.cast(x, tl.float16, fp_downcast_rounding="rtne")
y = al.cast(x, tl.int8,    overflow_mode="saturate")
```

**Cube-Vector pipeline sync:**
```python
# Cube side: signal Vector after tl.dot completes
al.sync_block_set(sender="cube", receiver="vector", event_id=0)  # event_id range 0-15

# Vector side: wait before processing accumulator output
al.sync_block_wait(sender="cube", receiver="vector", event_id=0)
```

**Multi-Vector Core post-dot parallelism:**
```python
SUB_BLK_M: tl.constexpr = BLOCK_M // 2
for s in al.parallel(0, 2, bind_sub_block=True):
    sub = al.extract_slice(acc, (s * SUB_BLK_M, 0), (SUB_BLK_M, BLOCK_N), (1, 1))
    sub = tl.where(sub > 0.0, sub, 0.0).to(tl.float16)
    tl.store(c_ptr + ..., sub)
```

**`num_warps` / `num_stages` on Ascend**: silently ignored. Use `NPUOptions` instead.

---

## Hardware-Specific Optimization

### Cube (AI Core)
- BLOCK_M/N/K multiples of 16 (512B / element size requirement)
- Accumulator in FP32
- Use diagonal scheduling for large matrices (L2 cache hit rate)

**Diagonal scheduling:**
```python
BLOCK_THRESHOLD: tl.constexpr = 4
for block_idx in range(pid, NUM_BLOCKS_M * NUM_BLOCKS_N, tl.num_programs(0)):
    if NUM_BLOCKS_M >= BLOCK_THRESHOLD and NUM_BLOCKS_N >= BLOCK_THRESHOLD:
        task_m = block_idx % NUM_BLOCKS_M
        task_n = (block_idx // NUM_BLOCKS_M) % NUM_BLOCKS_N
    else:
        task_m = block_idx // NUM_BLOCKS_N
        task_n = block_idx % NUM_BLOCKS_N
```

### UB (Vector Core)
- Total buffer size < 192KB including offset/mask/index arrays
- Single-value buffer 32B aligned

### Grid
- 1D Grid ≤ number of physical cores
- Intra-core loop processes multiple tiles
- Grid > 65535 causes silent crash

---

## Common Bottleneck Quick Reference

| cannsim Metric | Bottleneck | Typical Optimization |
|---------------|------------|---------------------|
| aiv_scalar > 80% | Scalar Bound | Check two-pass / per-row loop; change to single-pass `tl.sum(x,1)` |
| aiv_mte2 > 50% | Memory Bound | Contiguous memory access, expand+contiguous, increase BLOCK |
| aiv_vec > 50% | Compute Bound | Algorithm optimization, reduce redundant computation |
| aic_cube_ratio < 50% | Low Cube Utilization | Check alignment (512B), BLOCK multiples of 16, `compile_hint('dot_pad_only_k')` |

---

## Case Studies

### RoPE (npu_rotary_mul) — Contiguous Access Wins

| Version | Task Time | Bottleneck |
|---|---|---|
| per-row loop (div/mod) | 3605 µs | MTE3 95.6% (183K MTE2 ops) |
| 2D Tiling RPT=64 | 1362 µs | scalar 85% |
| Incremental pointer tracking | ~17800 µs | branch divergence (**anti-pattern**) |
| Contiguous access (expand) | **752 µs** | memory BW |

**Key insight**: `expand().contiguous()` on host eliminates broadcast stride entirely.
Contiguity > data reuse count. Even re-loading `cos`/`sin` with contiguous access
beats broadcasting with strided access.

**Pitfall**: 2D broadcast `row_off[:, None] + col_off[None, :]` appears non-contiguous to the
compiler even though each tile is 8KB. The compiler cannot merge these into large-block DMA.

### GroupNorm+Swish — Single-Pass Reduces 8.4× to 0.77×

| Version | Avg (µs) | vs torch_npu |
|---|---|---|
| Two-pass | 36.3 | 8.4× slower |
| Single pass | 3.3 | 0.77× (faster) |

Two-pass fix: load full row once when `NBLOCK = next_power_of_2(D)` fits in UB.
`tl.sum(x, 1)` performs axis=1 reduction; compiler keeps `x` in UB without re-reading.
NBLOCK up to 8192 is safe.

---

## al.multibuffer Pitfalls (verified June 2026)

`al.multibuffer(tensor, size=2)` is a **side-effect hint only** — do NOT reassign its return:

```python
# CORRECT: side-effect call, use original tensor variable
al.compile_hint(a, "dot_pad_only_k")  # compile_hint BEFORE multibuffer
al.compile_hint(b, "dot_pad_only_k")
al.multibuffer(a, size=2)  # side-effect only - no reassignment
al.multibuffer(b, size=2)
accumulator = tl.dot(a, b, accumulator)  # use ORIGINAL a, b

# WRONG: reassigning return value (returns None, crashes tl.dot and al.compile_hint)
a = al.multibuffer(a, size=2)   # a is now None -> CompilationError
al.compile_hint(a, "dot_pad_only_k")  # AttributeError: NoneType has no .handle'
```

Rule: `al.compile_hint` must be called **before** `al.multibuffer` on the same tensor.
UB budget for BLOCK_128x128x32 fp32: 96KB without double-buffering (OK within 65% factor).
Compiler manages ping-pong UB allocation internally, no manual UB accounting needed.

---

## Anti-Pattern Checklist (NEVER)

- Make optimization decisions based solely on single-scale data
- Optimize kernel directly when end-to-end target is not met (use cannsim first to confirm bottleneck)
- Sacrifice precision for performance / hardcode that breaks generalization
- Reduce directly in FP16 / matrix multiplication with BLOCK not multiple of 16
- BLOCK_SIZE exceeding UB (192KB) / non-contiguous memory access
- Use `tensor.item()` in hot path (triggers CPU-NPU synchronization)
- Use if branches inside loops to modify variables (Triton compiles to masked operations, catastrophic performance degradation)
- Calculate UB only for data buffer in 2D tiling (must include offset/mask/index arrays)
- Use precomputed offset tensors for 2D broadcasting (triggers compiler addptr multi-user assertion)
- Use broadcast stride to access auxiliary tensors (cos/sin, etc.) inside kernel — change to host-side expand+contiguous
- Two-pass mode for reduction operators — use single pass, compute everything within UB after one load
- Not using diagonal scheduling for large matrices (L2 cache thrashing, must enable above BLOCK_THRESHOLD)
- **ALWAYS gate persistent-grid dispatch on `cdiv(n, BLOCK_SIZE) > 65535` (see Rule 8). Applying persistent grid unconditionally is a regression at the bench shape.**

## Verification Checklist

- [ ] Precision aligns with PyTorch-NPU (rtol=1e-3, atol=1e-3)
- [ ] Non-aligned dimensions and boundaries pass
- [ ] Performance tests cover small/medium/large sizes
- [ ] grid ≤ number of physical cores, BLOCK_SIZE is compile-time constant
- [ ] Buffer < 192KB, all load/store have masks
- [ ] Reduction upcast to FP32, matrix multiplication BLOCK multiples of 16
- [ ] Is the reduction operator single-pass? (Required when D ≤ UB)
- [ ] Diagonal scheduling enabled for large matrices
- [ ] Elementwise kernel uses two-path dispatch (direct + persistent) — see Rule 8

## Constraints
- Always use `episode_retrieve` before applying any optimization pattern
- Always use `episode_write` after every successful optimization
- Precision (rtol=1e-3, atol=1e-3) is non-negotiable — roll back if not met
- Always verify with cannsim before declaring optimization complete
- For elementwise kernels, ALWAYS implement two-path dispatch (Rule 8) — never apply persistent grid unconditionally

---

## Full API & Compiler References

These files are part of this tree and must be consulted for full API details:

- **`../../shared/references/optimization-patterns.md`** (487 lines) — authoritative optimization reference; the patterns in this leaf are a subset. Contains:
  - Rule 3b: `care_padding=False` — ~5–10% free speedup when masked positions are unused downstream
  - `num_warps`/`num_stages` are **silently ignored** on Ascend NPU (no warp model); use NPUOptions instead
  - `next_power_of_2` trap: `triton.next_power_of_2(48) = 64` may exceed UB — use floor rounding when needed
  - Section 3 — Reference kernel implementations with full runnable code: GEMM (with diagonal grid scheduling + `al.parallel` multi-vector-core post-dot), LayerNorm, Online Softmax, Flash Attention
  - Section 4.1 pitfalls G3–G7: integer comparison in `tl.where` (cast to float32 first); multi-dim grid overhead; Cube never activated (`aic_cube_ratio ≈ 0` check); FP16 intermediate overflow; `.item()` CPU-NPU sync in hot path
  - RoPE case study: full 7-pitfall breakdown (MTE granularity, scalar overhead of 2D tiling, conditional branches catastrophic, precomputed offset table assertion, UB overflow from offset arrays, MTE instruction granularity, broadcast stride vs expand+contiguous)
  - GroupNorm+Swish case study: two-pass 36.3µs → single-pass 3.3µs (0.77× vs torch_npu 4.3µs); detailed `tl.sum(x, 1)` single-pass explanation

- **`./references/`** — 10 before/after operator implementations with perf measurements (use as lookup when working on the same operator type):
  - `l1_19_ReLU/` — ReLU: baseline + optimized (fp32 upcast fix)
  - `l1_23_Softmax/` — Softmax: baseline + optimized
  - `l1_26_GELU_/` — GELU: baseline + optimized
  - `l1_36_RMSNorm_/` — RMSNorm: baseline + optimized
  - `l1_41_Max_Pooling_1D/` — MaxPool1D: baseline + optimized
  - `l1_100_HingeLoss/` — HingeLoss: baseline + optimized
  - `l1_1_Square_matrix_multiplication_/` — Square MatMul: baseline + optimized
  - `l2_9_Matmul_Subtract_Multiply_ReLU/` — Matmul+Sub+Mul+ReLU: baseline + optimized
  - `l2_18_Matmul_Sum_Max_AvgPool_LogSumExp_LogSumExp/` — complex matmul fusion: baseline + optimized
  - `l2_76_Gemm_Add_ReLU/` — Gemm+Add+ReLU: baseline + optimized
  - Each directory: `{name}.py` (baseline), `opt_{name}.py` (optimized), `{name}_perf.txt`, `opt_{name}_perf.txt`

- **`./references/elementwise_two_path_dispatch.md`** — Detailed two-path dispatch pattern with the full implementation, threshold math, and the per-shape routing table (added June 2026 after the l1_5 matrix-scalar regression).

- **`../../shared/references/triton-api-reference.md`** — Complete Triton-Ascend API reference:
  - Section 4 (AL extension): Full enumerations (CORE, PIPE, MODE, FixpipeDMAMode, SYNC_IN_VF), all ops: `al.copy`, `al.fixpipe`, `al.debug_barrier`, `al.sync_block_set/wait`, `al.scope`, `al.custom`/`@register_custom_op`, math ops (`al.atan2`, `al.isfinited`), auxiliary ops (`al.parallel`, `al.compile_hint`, `al.multibuffer`), vector ops (`al.insert_slice`, `al.extract_slice`, `al.get_element`, `al.sort`, `al.flip`, `al.cast`), memory ops (`al.index_put`, `al.gather_out_to_ub`, `al.scatter_ub_to_out`, `al.index_select_simd`)
  - Section 5 (BL extension): `bl.alloc`, `bl.to_buffer`, `bl.to_tensor`, `bl.subview` — explicit on-chip memory management for advanced UB control
  - Section 7 (NPUOptions): All compiler flags: `sync_solver`, `enable_vf_fusion`, `vf_merge_level`, `add_auto_scheduling`, `enable_hivm_auto_cv_balance`, `enable_mixed_cv`, `multibuffer`, `enable_ubuf_saving`, `enable_preload`, `num_stages`, `mix_mode`, `compile_mode`, `debug`, `bisheng_options`, precision flags, layout flags, SIMT flags

- **`../../shared/references/tiling-strategies.md`** — Detailed tiling methodology:
  - Full UB budget calculation with all buffer types, alignment overhead, 2D tiling formula
  - All inter-core patterns with load balancing code, diagonal scheduling implementation
  - Operator case studies: LayerNorm, Softmax, MatMul with exact UB requirements

- **`../../shared/references/ascend-terminology.md`** — Hardware terminology:
  - HIVM IR mapping table, pipeline stages, memory hierarchy, alignment rules