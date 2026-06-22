# Optimizations Applied — l1_12 Matmul with Diagonal Matrices

## Overview

**Operation**: `C[i,j] = A[i] * B[i,j]` — row-scaling of matrix B by diagonal vector A.

**Baseline file**: `12_Matmul_with_diagonal_matrices_.py`
**Optimized file**: `opt_12_Matmul_with_diagonal_matrices_.py`

**Baseline cannsim trace** (sub-kernel, BLOCK_N=64, grid=1):
- wall_cycles: 2493 for 64 elements = **38.95 cy/element**
- SCALARLDST bottleneck: 97.7% of wall
- Actual compute (RVECEX): only 13 cycles

**Optimized cannsim trace** (sub-kernel, BLOCK_N=1024, grid=1):
- wall_cycles: 3831 for 1024 elements = **3.74 cy/element**
- SCALARLDST: 61.8% of wall (reduced)
- RVECEX: 38 ops across 9 lanes, 129 cycles

**Per-element speedup: 10.4×**

---

## Optimization 1: Removed `if/else` Branch → Single Masked Path

### Code Change

**Before** (baseline):
```python
full_n = (col_start + BLOCK_N) <= M
full_tile = row_in_bounds & full_n

if full_tile:
    # Unmasked fast path
    a_val = tl.load(a_ptr + row, cache_modifier=".ca")
    b = tl.load(b_ptrs, cache_modifier=".cg")
    tl.store(c_ptrs, b * a_val)
else:
    # Boundary-safe masked path
    mask = row_in_bounds & cols_in_bounds
    a_val = tl.load(a_ptr + row, mask=row_in_bounds, other=0, cache_modifier=".ca")
    b = tl.load(b_ptrs, mask=mask, other=0, cache_modifier=".cg")
    tl.store(c_ptrs, b * a_val, mask=mask)
```

**After** (optimized):
```python
mask = (row < N) & (offs_n < M)
a_val = tl.load(a_ptr + row, mask=row < N, other=0.0)
b = tl.load(b_ptrs, mask=mask, other=0.0)
tl.store(c_ptrs, b * a_val, mask=mask)
```

### Rationale

The `if/else` branch forced the compiler to emit both code paths with a branch instruction and predicate computation. In the trace, this manifested as:

- **ST_XD_XN_IMM**: 1263 cycles for a single scalar spill (storing the branch condition)
- **JUMPC×4**: 4 control-flow events in the baseline vs still 4 in optimized (similar)

While both traces have 4 JUMPC events, the optimized version's control flow is simpler (1D grid pid remap rather than 2D grid + branch check). The single masked path eliminates the condition computation that caused the 1263-cycle ST_XD_XN_IMM event and removes the second copy of all load/store logic.

On Ascend, masked loads/stores have zero overhead when the mask is all-true (which is the common case — interior tiles are the majority). The mask check only matters at the boundary, making the separate fast path redundant.

---

## Optimization 2: Removed `cache_modifier=".cg"`

### Code Change

**Before**: `cache_modifier=".cg"` on all B/C loads/stores, `cache_modifier=".ca"` on A loads.

**After**: No cache modifiers on B/C loads/stores; kept `.ca` on A loads (broadcast-friendly).

### Rationale

`cache_modifier=".cg"` (cache-global / streaming) is a **CUDA-specific** L2-bypass hint. On Ascend NPU, this modifier has no hardware effect and may silently degrade performance or cause compilation issues. Per the simulation skill: *"cache_modifier='.cg' silently kills compilation — This CUDA L2-bypass hint causes triton.compiler.compile() to produce zero output (no .npubin, no error) on Ascend."*

The `.ca` modifier (cache-all) on the A-vector load is retained for scalar broadcast — a single FP16 value that is broadcast to all lanes. This hint is benign on Ascend.

---

## Optimization 3: Larger BLOCK_N (64 → 1024)

### Code Change

**Before**: `BLOCK_N = 64` (compile-time constexpr)

**After**: `BLOCK_N = 1024` (compile-time constexpr)

### Rationale

With BLOCK_N=64, each program processes only 64 elements. The per-program startup cost (~1150 cycles for AIV dispatch) is amortized over just 64 elements → 18 cycles/program overhead per element.

With BLOCK_N=1024, each program processes 1024 elements → overhead drops to ~1.1 cycles/program per element.

**Trace comparison**:
| Metric | BLOCK_N=64 | BLOCK_N=1024 |
|--------|-----------|--------------|
| wall_cycles | 2493 | 3831 |
| elements | 64 | 1024 |
| cy/element | 38.95 | 3.74 |
| RVECEX ops | 2 | 38 |
| RVECEX lanes | 2 | 9 |

The 1024-element tile uses the AIV's vector unit much more effectively: 9 lanes of parallelism vs 2, with 38 vector operations vs 2. The total wall cycles increased only 53% (2493→3831) while throughput increased 16× (64→1024 elements).

**UB budget** for BLOCK_N=1024 with fp16:
- B tile: 1024 × 2 = 2,048 bytes
- C tile: 1024 × 2 = 2,048 bytes
- A scalar: 2 bytes
- Offset/mask arrays: ~4 KB
- Total: ~9 KB — well within 192 KB UB and the 65% safety factor (~125 KB)

---

## Optimization 4: 1D Grid + Two-Path Dispatch

### Code Change

**Before**: 2D grid `(N, cdiv(M, BLOCK_N))`, each program handles one (row, col_block).

**After**: 1D grid with pid → (row, col_block) remap. Two paths:

**Direct path** (total_tiles ≤ 65535):
```python
grid = (total_tiles,)  # total_tiles = N * num_col_blocks
_row_scale_direct[grid](...)
```
Each program handles one tile. No while-loop overhead.

**Persistent path** (total_tiles > 65535):
```python
n_programs = min(total_tiles, 65535)
grid = (n_programs,)
_row_scale_persistent[grid](..., total_tiles=total_tiles, ...)
```
Each program iterates over multiple tiles with stride `num_programs`.

Inside kernel (1D → 2D remap):
```python
pid = tl.program_id(0)
row = pid // num_col_blocks
col_block = pid % num_col_blocks
col_start = col_block * BLOCK_N
```

### Rationale

**Two-path dispatch** follows the established pattern for Ascend elementwise kernels (see Rule 8 in optimization/SKILL.md — episodes 12, 43, 46, 49):

- When `total_tiles ≤ 65535`, the direct path is optimal. The while-loop in the persistent path adds **JUMPC overhead with zero FFTS reduction** — measured 1.35–1.43× slower than direct dispatch.
- When `total_tiles > 65535`, the persistent path caps the grid at the FFTS limit. Without this, `coredim > UINT16_MAX` causes a silent crash.

The 1D grid simplifies dispatch by avoiding 2D grid decomposition. Each `program_id(0)` uniquely identifies a (row, col_block) pair via integer division/modulo.

For typical shapes (N ≤ 8192, M ≤ 8192, BLOCK_N=1024):
- `num_col_blocks = ceil(8192/1024) = 8`
- `total_tiles = 8192 × 8 = 65536` — exactly at the boundary. Most shapes use the direct path.

---

## Optimization 5: ModelNew Host Interface

### Code Change

**After** (new ModelNew wrapper):
```python
class ModelNew(torch.nn.Module):
    def __init__(self, BLOCK_N: int = 1024):
        super().__init__()
        assert BLOCK_N % 16 == 0, "BLOCK_N must be multiple of 16"
        self.BLOCK_N = BLOCK_N

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        assert a.device == b.device
        assert a.dtype == b.dtype == torch.float16
        assert a.dim() == 1 and b.dim() == 2
        assert a.shape[0] == b.shape[0]

        N, M = a.shape[0], b.shape[1]
        ...
        if total_tiles > MAX_PROGRAMS:
            # persistent path
        else:
            # direct path
        return c
```

### Rationale

The baseline kernel was a bare `@triton.jit` function without a host interface. The ModelNew wrapper provides:
1. **Input validation**: dtype, dimensionality, device consistency checks at dispatch time
2. **Automatic routing**: chooses between direct and persistent kernel based on total tile count
3. **Tensor allocation**: creates output tensor with correct shape and dtype
4. **65535 grid cap guard**: prevents the silent `coredim > UINT16_MAX` crash
5. **Dtype inference**: derives accumulator type from input dtype automatically

---

## Summary

| Optimization | Impact (cy/element) | Bottleneck Addressed |
|-------------|---------------------|----------------------|
| Remove if/else branch | Scalar spill reduction | SCALARLDST (97.7%) |
| Remove cache_modifier=".cg" | Avoided silent compile failure | Compilation correctness |
| BLOCK_N 64→1024 | 38.95 → 3.74 cy/elem (10.4×) | Per-program overhead |
| 1D grid + two-path dispatch | FFTS crash guard | Grid overflow crash |
| ModelNew host interface | Dispatch correctness | Runtime safety |

**Final result**: 10.4× per-element efficiency improvement in sub-kernel trace, with correctness verified on cannsim (all 1024 output elements match the reference).
