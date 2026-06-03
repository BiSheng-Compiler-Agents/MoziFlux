# PUSHQ / VF Instruction Dispatch Bottleneck in Matmul Kernels

## Discovery

Verified on l1_10 3D Tensor Matrix Multiplication (episode 55, June 2026, Ascend950).
The baseline kernel had PUSHQ as the dominant bottleneck at 86,982 busy_cyc (88% of wall),
driven by 2,056 VF (vector-format) instructions at an average of 72 cycles each.

## What Causes It

VF instructions represent vector instruction dispatch events. The count explodes when the
kernel generates many small vector operations per K-loop iteration:

| Source | VF events | % of PUSHQ |
|--------|-----------|------------|
| Mask computation (SHL/ADD_IMM/ZEROEXT) | ~4,100 | 49% |
| `a.to(tl.float32)` conversions (RV_VCVT_F2F) | ~2,304 | 28% |
| Pointer arithmetic (ST_XD_XN_IMM/LD_XD_XN_IMM spills) | ~162 events | 2% |
| JUMPC loop control flow | ~4,153 events | — |

Together these produce **2,056 VF events** in the baseline (BLOCK_K=32, K=64, 2 iterations).

## How to Fix (Measured Reduction: 2,056 → 8 VF events)

### 1. Mask hoisting (biggest win)

### V2 Update — Further reduction from 14,141 to 8,689 cycles (June 2026)

The V1 fixes above reduced wall_cycles from 98,817 to 14,141 (VF 2,056→8).
V2 applies two additional optimizations that cut another 38.5%, reaching 8,689 cycles:

**5. `tl.dot(a, b, acc)` in-place accumulation:** Eliminates the 64 KB fp32 temporary tile
by accumulating inside the Cube hardware. This alone cuts RVECEX from 1,219 to 24
busy_cyc (−98%) and reduces RV_VSTI/RV_VLDI by 33-38%.

**6. `tl.range` instead of `while`:** A `while k_iter < K` loop inserts dynamic exit
checks the compiler cannot pipeline. `for k_idx in tl.range(0, num_k_iters)` tells the
compiler the exact trip count, enabling pipelining. Halves FLOWCTRL (9,105→3,880 busy_cyc),
reduces JUMPC from 55 to 50 (in V1, JUMPC was already reduced; in V2 baseline vs V2 opt:
JUMPC went from 5,180 to 50).

**7. Remove `al.multibuffer`:** V1 used `al.multibuffer(size=2)` which adds UB overflow
risk on real hardware. In-place dot makes multibuffer redundant (the 64 KB temp it was
overlapping is gone). V2 deletes multibuffer entirely.

| Metric | V1 (14,141 cyc) | V2 (8,689 cyc) | Δ |
|--------|-----------------|----------------|-----|
| VF events | 8 | 4 | −50% |
| RVECEX busy_cyc | 1,219 | 24 | −98.0% |
| RV_VSTI ops | 3,072 | 2,048 | −33.3% |
| RV_VLDI ops | 3,328 | 2,048 | −38.5% |
| FLOWCTRL busy_cyc | 9,105 | 3,880 | −57.4% |
| SET_INTRA_BLOCKI avg | 913 cyc | 518 cyc | −43.3% |

**Combined rule:** When using `tl.dot(a,b,acc)` in-place AND `tl.range`, delete any
`al.multibuffer` lines. The in-place dot gives 10× the gain of double buffering with
zero UB overflow risk.

---

### 1. Mask hoisting (biggest win)
Hoist M-dim and N-dim masks outside the K loop. The K-dim portion of the mask is cheap
(a single `k_offs[N,:] < K` per iteration). This eliminates ~4,100 scalar ops per loop.

**Before:**
```python
while k_iter < K:
    a_mask = (offs_m[:, None] < M) & (k_iter + offs_k[None, :] < K)
    b_mask = (k_iter + offs_k[:, None] < K) & (offs_n[None, :] < N)
```

**After:**
```python
m_mask = offs_m[:, None] < M   # hoisted
n_mask = offs_n[None, :] < N   # hoisted
while k_iter < K:
    k_offs = k_iter + offs_k
    a_mask = m_mask & (k_offs[None, :] < K)   # only K-varies
    b_mask = (k_offs[:, None] < K) & n_mask
```

### 2. Remove dtype conversions before tl.dot
`a.to(tl.float32)` before `tl.dot` generates 2,304 VCVT_F2F instructions (16,128 total cycles).
More critically, it causes **ALL output elements to mismatch** (max_rel_err=1.0) — the
tl.dot fp32 accumulator expects fp16 inputs natively.

**Before:** `acc += tl.dot(a.to(tl.float32), b.to(tl.float32))`
**After:**  `acc += tl.dot(a, b)`

### 3. Increase BLOCK_K
Larger BLOCK_K means fewer K-loop iterations for the same K, reducing all per-iteration
overhead proportionally. BLOCK_K=64 (vs 32) halves iteration count.

### 4. Recompute pointers vs increment
Instead of `a_ptrs += BLOCK_K * stride_ak` inside the loop, recompute from base + offset
each iteration. This eliminates the pointer-update instruction chain and reduces SCALARLDST
spills.

## Sub-kernel Trace Comparison

| Metric | Baseline | After Fix | Reduction |
|--------|----------|-----------|-----------|
| VF (PUSHQ) events | 2,056 | 8 | 99.6% |
| wall_cycles | 98,817 | 14,141 | 7.0× |
| SHL ops | 4,163 | 0 | 100% |
| ADD_IMM ops | 4,113 | 0 | 100% |
| ZEROEXT ops | 4,102 | 0 | 100% |
| RV_VCVT_F2F | 2,304 | 0 | 100% |
| JUMPC | 4,153 | 55 | 98.7% |
| SCALAR busy_cyc | 27,268 | 1,766 | 93.5% |

## When to Suspect PUSHQ/VF Bottleneck

Check your trace for:

```
10_PUSHQ        NNNN     XXXXX     XXXXX      X  ← BOTTLENECK
VF               2056     148512       72      ← CRITICAL
```

If PUSHQ busy_cyc > 50% of wall_cycles and VF is the top CRITICAL instruction,
you have a dispatch-pressure problem. The fix is always: **reduce per-iteration work,
not increase throughput per operation**.
