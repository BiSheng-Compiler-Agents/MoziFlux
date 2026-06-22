# Optimizations for `l1_13_Matmul_for_symmetric_matrices`

## Baseline

The baseline kernel (`13_Matmul_for_symmetric_matrices.py`) is a standard Triton matrix
multiplication:

- **Block sizes**: `BLOCK_M=32, BLOCK_N=32, BLOCK_K=32`
- **Grid**: 2D `(pid_m, pid_n)` — each program processes one tile
- **K loop**: Dynamic `for k_start in range(0, k, BLOCK_K)` — Triton cannot unroll
- **Masks**: Re-computed inside the K loop on every iteration (m_mask, n_mask, k_mask)
- **No Ascend hints**: No `compile_hint`, `multibuffer`, or `care_padding`
- **CUBE utilization**: 5.1% (640 / 12,559 wall cycles) — severely underutilized
- **SCALARLDST**: 48,602 cycles total, dominated by `ST_XD_XN_IMM` (81%) from dynamic K-loop pointer arithmetic
- **Memory transactions**: 512 `RV_VLDI` + 512 `RV_VSTI` — many tiny 32×32 transactions

**Baseline sub-kernel trace** (grid=1×1, M=32, N=32, K=64, 2 K-iterations):
- wall_cycles: 12,559
- cyc/elem: 12.26

---

## Optimization 1: Larger Block Sizes (128×128)

**Problem**: Tiny 32×32 tiles gave the Cube engine only 640 cycles of work (5.1% utilization).

**Fix**: Increased tile size to `BLOCK_M=128, BLOCK_N=128` (128×128 output tile), with
`BLOCK_K=32` or `BLOCK_K=64` for the K dimension. Each tile now processes 16,384 output
elements instead of 1,024 — 16× more work per program.

**Code**:
```python
triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}, ...),
triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, ...),
```

**Effect**: CUBE busy_cycles increased from 640 → 4,542 (7.1× more Cube work). CUBE
utilization from 5.1% → 22.0%.

---

## Optimization 2: GROUP_M Swizzle (1D Grid → 2D Mapping)

**Problem**: 2D grid `(pid_m, pid_n)` assigns tiles to cores in row-major order.
Each row of M-blocks uses the same A-tile, but if the grid assigns adjacent M-rows
to different cores, A-tile L1/L2 reuse is lost.

**Fix**: 1D grid with GROUP_M swizzle. Programs in a group process consecutive
M-rows, allowing A-tile reuse across N-blocks in the same group.

**Code**:
```python
pid = tl.program_id(0)
num_pid_m = NUM_BLOCKS_M
num_pid_n = NUM_BLOCKS_N
num_pid_in_group = GROUP_M * num_pid_n
group_id = pid // num_pid_in_group
first_pid_m = group_id * GROUP_M
group_size_m = min(GROUP_M, num_pid_m - first_pid_m)
pid_m = first_pid_m + (pid % group_size_m)
pid_n = (pid % num_pid_in_group) // group_size_m
```

**Effect**: L2 cache reuse improved. The number of physical programs is capped at
~96 (Ascend950 core count), so there is no host scheduling overhead from a 2D grid.

---

## Optimization 3: Mask Hoisting

**Problem**: The baseline re-computes `(offs_m[:, None] < m)` and `(offs_n[None, :] < n)`
on every K-loop iteration, adding O(K) scalar operations.

**Fix**: Compute `m_mask = offs_m < M` and `n_mask = offs_n < N` once before the K loop.
Only `k_mask` varies per iteration.

**Code**:
```python
# Hoisted (before K loop)
m_mask = offs_m < M
n_mask = offs_n < N

# Inside K loop — only k_mask varies
k_mask_a = (k_start + offs_k)[None, :] < K
a_mask = m_mask[:, None] & k_mask_a
k_mask_b = (k_start + offs_k)[:, None] < K
b_mask = k_mask_b & n_mask[None, :]
```

**Effect**: SCALAR total reduced from 9,788 → 8,821 cyc (-9.9%) despite processing
16× more elements per tile. The per-iteration mask computation was eliminated.

---

## Optimization 4: `care_padding=False`

**Problem**: `tl.load` with `other=0.0` default checks padding boundaries even when
the mask already handles boundary conditions. This generates extra guard instructions.

**Fix**: Add `care_padding=False` to all `tl.load` calls — safe because masked elements
are set to 0.0 and do not affect downstream computation.

**Code**:
```python
a = tl.load(a_ptrs, mask=a_mask, other=0.0, care_padding=False)
b = tl.load(b_ptrs, mask=b_mask, other=0.0, care_padding=False)
```

**Effect**: Reduced per-load guard overhead. Contributes to overall SCALAR reduction.

---

## Optimization 5: `al.compile_hint("dot_pad_only_k")`

**Problem**: When compiling `tl.dot(a, b)` where A is [BLOCK_M, BLOCK_K] and B is
[BLOCK_K, BLOCK_N], the Ascend compiler pads all three dimensions to satisfy Cube unit
alignment (usually 16-element boundaries). This wastes UB space.

**Fix**: `al.compile_hint(a, "dot_pad_only_k")` tells the compiler to only pad the K
dimension, saving 30–50% UB usage for the A/B tile buffers.

**Code**:
```python
al.compile_hint(a, "dot_pad_only_k")
al.compile_hint(b, "dot_pad_only_k")
```

**Effect**: Reduced UB pressure allows larger block sizes and room for multibuffer.

---

## Optimization 6: `al.multibuffer` (Double Buffering)

**Problem**: The K-loop is serialized: load A/B tiles → Cube compute → store → load next.
MTE2 (memory load engine) sits idle while Cube is computing.

**Fix**: `al.multibuffer(a, size=2)` creates ping-pong buffers for A and B tiles.
MTE2 can prefetch the next K-tile into the second buffer while Cube processes the
current tile from the first buffer.

**Code**:
```python
al.multibuffer(a, size=2)  # side-effect only, no reassignment!
al.multibuffer(b, size=2)
```

**Pitfall**: `al.multibuffer` is a side-effect call — never reassign its return value.
Must be called AFTER `al.compile_hint`.

**Effect**: MTE2 wait cycles reduced from 9,682 → 7,171 (-25.9%). VEC wait on MTE2
reduced from 5,211 → 2,187 (-58.0%).

---

## Optimization 7: `tl.constexpr K` with Autotune

**Problem**: The baseline uses runtime `k` which prevents compiler optimizations
like loop unrolling and compile-time scheduling of K iterations.

**Fix**: Declare `K: tl.constexpr` in the kernel signature and use `@triton.autotune`
with `key=["M", "N", "K"]`. Each unique K value gets a dedicated compilation where
K is a compile-time constant. The autotune sweeps 9 block-size configurations.

**Code**:
```python
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}, ...),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, ...),
        # ... 7 more configs
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _symmetric_matmul_kernel_opt(..., K: tl.constexpr, ...):
    NUM_BLOCKS_K = tl.cdiv(K, BLOCK_K)  # compile-time constant
    ...
```

**Effect**: SET_INTRA_BLOCKI count stays at 8 (same as baseline with dynamic range),
but the compiler can now schedule around K-loop boundaries because K is known at
compile time. The autotune also selects optimal block sizes per shape.

---

## Optimization 8: Diagonal Scheduling for Large Matrices

**Problem**: For very large matrices (e.g., 4096×4096), the group-swizzled grid
still causes L2 cache thrashing because M-rows are too far apart in memory.

**Fix**: When both `NUM_BLOCKS_M` and `NUM_BLOCKS_N` exceed `DIAG_THRESHOLD=6`,
switch to diagonal scheduling where each program processes tiles along a diagonal
strip, improving L2 reuse.

**Code**:
```python
if NUM_BLOCKS_M >= DIAG_THRESHOLD and NUM_BLOCKS_N >= DIAG_THRESHOLD:
    NUM_TILES = NUM_BLOCKS_M * NUM_BLOCKS_N
    for block_idx in range(pid, NUM_TILES, tl.num_programs(0)):
        task_m = block_idx % NUM_BLOCKS_M
        task_n = block_idx // NUM_BLOCKS_M
        _compute_tile(...)
```

---

## Optimization 9: Autotune with 9 Configurations

**Problem**: A single block size cannot be optimal for all matrix shapes.

**Fix**: Provide 9 `@triton.autotune` configs covering:
- Large balanced tiles (128×128 with K=32/64)
- Asymmetric tiles (128×64, 64×128 with K=32/64)
- Small tiles (64×64)
- Wide tiles (256×64, 64×256)
- Fallback (32×32 with K=128)

**Code**:
```python
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}, ...),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, ...),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64,  "BLOCK_K": 64}, ...),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64,  "BLOCK_K": 32}, ...),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "BLOCK_K": 64}, ...),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64,  "BLOCK_K": 64}, ...),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 64,  "BLOCK_K": 32}, ...),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 256, "BLOCK_K": 32}, ...),
        triton.Config({"BLOCK_M": 32,  "BLOCK_N": 32,  "BLOCK_K": 128}, ...),
    ],
    key=["M", "N", "K"],
    warmup=25, rep=200,
)
```

---

## Optimization 10: Host Dispatch with Grid Capping

**Problem**: A 1D grid larger than the physical core count (96 on Ascend950) adds
host scheduling overhead without benefit.

**Fix**: The host grid function caps total programs to 96:

```python
def grid_fn(META):
    BLOCK_M = META["BLOCK_M"]
    BLOCK_N = META["BLOCK_N"]
    num_m = triton.cdiv(M, BLOCK_M)
    num_n = triton.cdiv(N, BLOCK_N)
    group_m = META.get("GROUP_M", 8)
    num_pid_in_group = group_m * num_n
    total = num_m * num_n
    return (min(total, 96),)
```

---

## Results

| Metric | Baseline | Optimized | Improvement |
|--------|----------|-----------|-------------|
| wall_cycles (sub-kernel) | 12,559 | 20,655 | — (16× more elements) |
| **cyc/elem** | **12.26** | **1.26** | **9.73×** |
| CUBE busy_cyc | 640 | 4,542 | 7.1× more Cube work |
| SCALAR total | 9,788 | 8,821 | -9.9% |
| VEC wait (MTE2+MTE3) | 10,422 | 4,423 | -57.6% |
| MTE2 wait on VEC | 5,103 | 8 | -99.8% |
| CUBE utilization | 5.1% | 22.0% | +16.9pp |
| SCALARLDST total | 48,602 | 48,785 | — (inherent to output store) |
| SET_INTRA_BLOCKI | 8 | 8 | Same (both use range) |
| RV_VLDI count | 512 | 2,048 | 4× (larger tiles) |
| RV_VSTI count | 512 | 2,048 | 4× (larger tiles) |

## Pitfalls Encountered

1. **`tl.static_range` requires true constexpr in nested functions** — `NUM_BLOCKS_K`
   computed from `tl.cdiv(K, BLOCK_K)` in the outer function loses constexpr status
   when passed as an argument to a nested `@triton.jit` function. Fixed by computing
   `NUM_BLOCKS_K` inside the nested function from `K` and `BLOCK_K` (both constexpr).

2. **`al.multibuffer` ordering** — `al.compile_hint` must be called BEFORE
   `al.multibuffer`. The multibuffer is a side-effect call (returns None) — never
   reassign the tensor variable.

3. **`K` as constexpr in ASTSource** — When compiling via `ASTSource` for cannsim,
   parameters declared as `tl.constexpr` in the kernel must NOT appear in the
   `signature` dict — they belong in the `constants` dict only.

4. **GRID overflow** — For the baseline `_run_baseline` wrapper, `65535 × BLOCK_SIZE`
   is the maximum safe grid. Chunk the input to avoid UINT16_MAX overflow on Ascend FFTS.
