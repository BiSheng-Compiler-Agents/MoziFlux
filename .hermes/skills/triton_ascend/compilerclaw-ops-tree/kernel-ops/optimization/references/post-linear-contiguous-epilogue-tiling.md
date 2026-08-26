# Post-linear contiguous epilogue tiling

Use this when a KernelBench-style operator is `nn.Linear`/GEMM followed by a pure elementwise epilogue (scale, clamp/Hardtanh, GELU, Swish, etc.) and the GEMM output is contiguous.

## Pattern

If `y = self.gemm(x)` is contiguous, prefer a flat 1D epilogue over `(row, column-block)` tiling:

```python
_BLOCK_SIZE = 4096
_MAX_PROGRAMS = 65535

n_elements = y.numel()
n_tiles = triton.cdiv(n_elements, _BLOCK_SIZE)
if n_tiles > _MAX_PROGRAMS:
    _epilogue_persistent[(_MAX_PROGRAMS,)](y, y, n_elements, _MAX_PROGRAMS, ..., BLOCK_SIZE=_BLOCK_SIZE)
else:
    _epilogue_direct[(n_tiles,)](y, y, n_elements, ..., BLOCK_SIZE=_BLOCK_SIZE)
```

Kernel loop:

```python
offs = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
mask = offs < n_elements
x = tl.load(ptr + offs, mask=mask, other=0.0, care_padding=False)
# fp32 epilogue math, cast back
tl.store(ptr + offs, out.to(x.dtype), mask=mask)
```

## Why it helps

Row/block tiling can overlaunch programs for large matrices. Example target `(M,N)=(2048,8192)` with `BLOCK_N=1024` launches `2048*8=16384` epilogue programs; flat `BLOCK_SIZE=4096` launches `4096` programs. The flat layout also removes row-stride arithmetic and makes contiguous MTE access obvious.

For scalar division epilogues (`ReLU(x) / divisor`, scaling, normalization by a constant), compute `scale = 1.0 / divisor` once in the host and use `x * scale` in the Triton kernel. This replaces expensive vector divide instructions with vector multiply while preserving semantics for invariant nonzero divisors.

## Cannsim comparison rule

When changing epilogue block size, sub-kernel wall cycles are not directly comparable because one trace may process more elements. Compare normalized throughput:

```text
elements_per_cycle = elements_in_subkernel / wall_cycles
```

Record both absolute cycles and normalized cycles/element in `performance_report.md`.

## Correctness and dispatch coverage

- Preserve exact activation semantics unless the task allows approximation (e.g. exact GELU should keep `erf`, not tanh approximation).
- Keep a persistent fallback only for `cdiv(n_elements, BLOCK_SIZE) > 65535`; do not use persistent unconditionally.
- In `profile_kernels.py`, force-test the persistent path by temporarily lowering `_MAX_PROGRAMS` rather than allocating huge tensors.
- Remove source-level cache modifiers that are not portable on Ascend shims; do not report the shim as bit-for-bit source if it differs for compile legality.
