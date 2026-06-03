# Optimizations: l1_2 Standard Matrix Multiplication

## Baseline Issues

The baseline kernel (`2_Standard_matrix_multiplication_.py`) had five key problems:

1. **2D grid (cdiv(M,32), cdiv(N,32))** — produces up to 16,384+ programs for large matrices,
   far exceeding the 32 physical cores. Massive FFTS dispatch overhead on every launch.

2. **BLOCK_M=32, BLOCK_N=32, BLOCK_K=32** — tiles too small for efficient Cube utilization.
   cannsim sub-kernel trace: CUBE only 640 busy_cyc out of 8,919 wall_cycles (~7.2%).

3. **Mask expressions recomputed inside K loop** — `(offs_m[:, None] < M)` and
   `(offs_n[None, :] < N)` don't change per K iteration but are anded fresh each step,
   adding scalar overhead.

4. **No `compile_hint("dot_pad_only_k")`** — bishengir pads all three dims (M, N, K),
   wasting Cube scheduling cycles and UB space.

5. **Dynamic K loop** — `for k_start in range(0, K, BLOCK_K)` is a runtime loop.
   Compiler cannot pipeline across K iterations. SET_INTRA_BLOCKI sync points per tile = 8.

---

## Optimizations Applied

### 1. 1D Grid with GROUP_M=4 Pid Swizzle

Replaces the 2D grid with a 1D grid of `num_pid_m * num_pid_n` programs, reordered
via the GROUP_M swizzle to improve L2 cache reuse:

```python
group_width  = GROUP_M * NUM_PID_N       # e.g. 4 * num_pid_n
group_id     = pid // group_width
first_pid_m  = group_id * GROUP_M
group_size_m = tl.minimum(NUM_PID_M - first_pid_m, GROUP_M)
pid_in_group = pid % group_width
pid_m        = first_pid_m + (pid_in_group % group_size_m)
pid_n        = pid_in_group // group_size_m
```

Effect: Adjacent programs share A-tile rows (same pid_m band), maximizing L2 hits.
Dispatch overhead eliminated — 1D grid maps directly to physical core scheduler.

### 2. Larger Tile Sizes: BLOCK_M=128, BLOCK_N=128, BLOCK_K=32

UB budget (fp32):
- A tile:  128 × 32 × 4 B = 16 KB
- B tile:  32 × 128 × 4 B = 16 KB
- Acc:     128 × 128 × 4 B = 64 KB
- Total:   96 KB  << 192 KB limit (50% utilization)

cannsim: CUBE busy_cyc 640 (7.2%) → 4,540 (26.9%) — 7× improvement.
tl.multiple_of / tl.max_contiguous hints added so compiler can emit aligned loads.

### 3. Row/Col Masks Hoisted Outside K Loop

```python
m_mask = offs_m < M   # computed once
n_mask = offs_n < N   # computed once
for _ in tl.static_range(NUM_K_TILES):
    k_mask = (k_off + offs_k) < K
    a_mask = m_mask[:, None] & k_mask[None, :]   # cheap AND only
    b_mask = k_mask[:, None] & n_mask[None, :]
```

Eliminates per-iteration scalar comparison on M/N dimensions.

### 4. `al.compile_hint(a/b, "dot_pad_only_k")`

BLOCK_M=128 and BLOCK_N=128 are cube-aligned (multiples of 16). Only K=32 may need
padding. Telling bishengir this reduces UB allocation ~30-50% vs padding all three dims.

```python
al.compile_hint(a, "dot_pad_only_k")
al.compile_hint(b, "dot_pad_only_k")
```

### 5. `care_padding=False` on All Loads

Padding values are not used downstream (they get masked before `tl.dot`). Skipping
the padding check saves ~5-10% per load at no correctness cost.

```python
a = tl.load(a_ptrs, mask=a_mask, other=0.0, care_padding=False)
b = tl.load(b_ptrs, mask=b_mask, other=0.0, care_padding=False)
```

### 6. `al.multibuffer(a/b, size=2)` — Double-Buffering

Allocates ping-pong A/B UB buffers so MTE2 can prefetch the next K-tile while CUBE
processes the current one. This is a **side-effect hint** — do NOT reassign its return:

```python
al.compile_hint(a, "dot_pad_only_k")   # must precede multibuffer
al.compile_hint(b, "dot_pad_only_k")
al.multibuffer(a, size=2)              # side-effect only — do NOT do: a = al.multibuffer(...)
al.multibuffer(b, size=2)
accumulator = tl.dot(a, b, accumulator)
```

cannsim: MTE2 busy_cyc 5,844 → 2,420 (2.4× reduction).

### 7. `tl.static_range(NUM_K_TILES)` with `NUM_K_TILES: tl.constexpr`

Fully unrolls the K loop at compile time. Allows bishengir to see all K iterations
simultaneously and insert software pipelining across them.

```python
NUM_K_TILES: tl.constexpr  # = triton.cdiv(K, BLOCK_K)

for _ in tl.static_range(NUM_K_TILES):
    k_off  = _ * BLOCK_K
    k_mask = (k_off + offs_k) < K
    ...
    accumulator = tl.dot(a, b, accumulator)
    a_ptrs += BLOCK_K * stride_ak
    b_ptrs += BLOCK_K * stride_bk
```

cannsim: SET_INTRA_BLOCKI count 8 → 2 (4× reduction), FLOWCTRL 9,707 → 6,829 cycles.

---

## Results

| Metric                        | Baseline  | Opt v2 (this kernel) |
|-------------------------------|-----------|----------------------|
| cyc/output element (cannsim)  | 8.71      | **0.95** (9.2× better) |
| CUBE utilization              | 7.2%      | 29.2%                |
| MTE2 busy_cyc (sub-kernel)    | 3,328     | 2,420                |
| FLOWCTRL busy_cyc             | 5,128     | 6,829 (more work/tile)|
| SET_INTRA_BLOCKI count        | 8         | 2                    |

Remaining bottleneck: MTE3 (output store, C tiles → GM) at 10,551 busy_cyc = 68%
of wall_cycles. This is inherent to any matmul output write and is hard to reduce.

---

## Pitfalls Found

- `al.multibuffer` returns `None` — reassigning its result crashes `tl.dot` and
  `al.compile_hint` with `AttributeError: NoneType has no attribute 'handle'`.
- `al.compile_hint` must be called **before** `al.multibuffer` on the same tensor.
- `tl.static_range` loop variable `_` gives the iteration index (0, 1, …, N-1);
  compute `k_off = _ * BLOCK_K` and still advance `a_ptrs`/`b_ptrs` at loop bottom.
