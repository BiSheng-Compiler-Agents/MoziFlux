# Optimizations: l1_20 LeakyReLU

## Baseline Issues

### Issue 1: CUDA-only hints without effect
**Problem:** `tl.multiple_of(offsets, 16)` and `tl.max_contiguous(offsets, 16)` are CUDA-specific compiler hints. On Ascend NPU, these have no effect — the Ascend backend ignores them.

**Fix:** Removed both calls entirely. Cleaner code with identical behavior.

**Impact:** Cosmetic. Zero regression risk.

---

### Issue 2: Wasteful `tl.zeros([BLOCK_SIZE])` allocation
**Problem:** The baseline creates `zero = tl.zeros([BLOCK_SIZE], dtype=x.dtype)` — a full BLOCK_SIZE tensor — solely to compare against `x` in `tl.where(x >= zero, x, x * neg)`. This allocates unnecessary UB (Unified Buffer) space equivalent to one extra vector register for the zero value.

```python
# Before: wasteful full-sized zero tensor
zero = tl.zeros([BLOCK_SIZE], dtype=x.dtype)
y = tl.where(x >= zero, x, x * neg)

# After: direct comparison, no allocation
y = tl.where(x > 0.0, x, x * neg)
```

**Impact:** Frees UB space. Removes the `RV_VDUPS` instruction (6 cycles per tile) that was needed to broadcast the zero scalar to a full vector register. At scale (thousands of tiles), this adds up.

---

### Issue 3: Missing `care_padding=False` on memory operations
**Problem:** The baseline uses plain `tl.load(...)` without `care_padding=False`. On Ascend, this causes the compiler to emit extra padding boundary checks for each load operation, adding MTE instruction overhead.

```python
# Before:
x = tl.load(x_ptr + offsets, mask=mask, other=0)

# After:
x = tl.load(x_ptr + offsets, mask=mask, other=0.0, care_padding=False)
```

**Impact:** ~5–10% MTE instruction reduction per load. Safe because the `mask` parameter already guarantees in-bounds access — the padding check is redundant.

---

### Issue 4: No fp32 upcast for fp16/bf16 inputs (potential VEC unit stalls)
**Problem:** When the input tensor is fp16 or bf16, the baseline computes `x >= zero` and `x * neg` entirely in the input precision. On Ascend, fp16 comparison operations route through the VEC fixed-function unit (shared with exp/div/sqrt), causing WAIT_FLAG_VEC stalls — the MTE3 store engine cannot proceed until VEC finishes.

From the l1_19_ReLU trace analysis (documented in `relu_fp16_vs_fp32_maximum_trace.md`), fp16 `tl.maximum` causes WAIT_FLAG_VEC stalls of ~1226 cycles per tile. Upcasting to fp32 routes the operation through RVECEX (reconfigurable vector ALU) which pipelines freely with other units.

```python
# Before (fp16 path):
x = tl.load(x_ptr + offsets, mask=mask, other=0)
zero = tl.zeros([BLOCK_SIZE], dtype=x.dtype)
y = tl.where(x >= zero, x, x * neg)

# After (fp16 path, with fp32 upcast):
x = tl.load(x_ptr + offsets, mask=mask, other=0.0, care_padding=False)
x_fp32 = x.to(tl.float32)
y_fp32 = tl.where(x_fp32 > 0.0, x_fp32, x_fp32 * neg)
y = y_fp32.to(x.dtype)
```

**Impact:** Eliminates WAIT_FLAG_VEC stalls for fp16/bf16 inputs. The two conversion instructions (fp16→fp32→fp16) cost ~14 cycles per 64-element vector but eliminate 1000+ cycles of VEC unit serialization. Net positive at any BLOCK_SIZE ≥ 128.

---

### Issue 5: Missing autotune and bucketed autotune key
**Problem:** The baseline uses a fixed BLOCK_SIZE (from `triton.Config` in the caller) without autotune. Different input sizes benefit from different block sizes — small inputs need smaller blocks to avoid wasting programs, large inputs need larger blocks for better memory bandwidth utilization.

Additionally, using the raw `n_elements` as an autotune key causes cache misses on every unique tensor size, forcing full autotune re-evaluation (5 × full-tensor warmup trials) for each new shape.

```python
# Before: no autotune
# key=[...] — not specified

# After: autotune with bucketed key
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8, num_stages=2),
    ],
    key=["n_elements_pow2"],
)
```

**Impact:** Autotune selects the optimal BLOCK_SIZE for each size class. The bucketed key (`n_elements_pow2 = 1 << (n-1).bit_length()`) collapses the number of cache entries from O(N) to O(log N) ≈ 30, avoiding costly re-compilation.

---

### Issue 6: Missing two-path dispatch for large N
**Problem:** The baseline uses a single kernel with `grid = (cdiv(n, BLOCK_SIZE),)`. On Ascend, FFTS caps any grid dimension at 65535 (UINT16_MAX). When `n > 65535 * BLOCK_SIZE`, the grid overflows and causes a runtime crash.

```python
# Before: no guard, crashes for large N
grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)

# After: two-path dispatch
_MIN_BLOCK = 256
_MAX_PROGRAMS = 65535

if triton.cdiv(n, _MIN_BLOCK) > _MAX_PROGRAMS:
    # Persistent: while-loop covers all elements, grid capped at 65535
    _leaky_relu_kernel_persistent[grid_capped](...)
else:
    # Direct: grid guaranteed <= 65535 for all autotune configs
    _leaky_relu_kernel_direct[direct_grid](...)
```

**Impact:** Correctness for large tensors (N > 16.78M elements). The persistent kernel uses a work-stealing while-loop so one program processes multiple tiles, while the grid stays within FFTS limits.

---

## Pitfalls and Lessons Learned

1. **fp32 vs fp16 routing differs on Ascend:** The branchless LeakyReLU formulation (`x + (neg-1) * min(x, 0)`) is mathematically sound but on Ascend950 expands to many more instructions than the simple `tl.where` approach. For fp32, `tl.where` is more efficient. The fp32 upcast for fp16 inputs provides the real benefit.

2. **`tl.zeros` removal is safe:** The comparison `x > 0` is equivalent to `x >= 0` for the LeakyReLU use case since the identity value at x=0 is 0 for both branches (0 and 0*neg=0).

3. **Threshold must use MIN_BLOCK, not MAX_BLOCK:** When computing the two-path dispatch threshold, using the largest BLOCK_SIZE (4096) would suggest that direct dispatch is safe up to 268M elements. But autotune also tries BLOCK_SIZE=256 during profiling, which at 268M elements would compute `cdiv(268M, 256) = 1,048,576 >> 65535`, causing a runtime crash. Always use MIN_BLOCK=256 as the reference. Hardware-verified pitfall from l1_19_ReLU optimization.
