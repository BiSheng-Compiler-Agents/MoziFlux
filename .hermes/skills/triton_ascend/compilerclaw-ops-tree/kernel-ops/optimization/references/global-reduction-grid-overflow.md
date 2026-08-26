# Global Reduction Grid-Overflow Pattern

Use for whole-tensor reductions followed by an elementwise pass (e.g. Frobenius norm `y = x / sqrt(sum(x*x))`) when the direct per-block grid may exceed Ascend `coreDim <= 65535`.

## Pattern

Keep two dispatch paths:

```python
n_tiles = triton.cdiv(n_elements, BLOCK)
if n_tiles <= 65535:
    _sumsq_atomic[(n_tiles,)](x, n_elements, sumsq, BLOCK=BLOCK)
else:
    n_programs = 65535
    partials = torch.empty((n_tiles,), device=x.device, dtype=torch.float32)
    _partial_sumsq[(n_programs,)](x, partials, n_elements, n_tiles, n_programs, BLOCK=BLOCK)
    _reduce_partials[(1,)](partials, sumsq, n_tiles, BLOCK=REDUCE_BLOCK)

_scale[(min(n_tiles, 65535),)](x, y, n_elements, sumsq, n_tiles, min(n_tiles, 65535), BLOCK=BLOCK)
```

Device-side persistent loop must iterate over **tiles**, not elements:

```python
tile = tl.program_id(0)
while tile < n_tiles:
    idx = tile * BLOCK + tl.arange(0, BLOCK)
    mask = idx < n_elements
    ...
    tile += n_programs
```

## Why two paths

The partial-reduction path adds one launch and a compact partial scan, so it can regress at sub-kernel or small-shape scale. It is still required for target shapes whose direct grid is illegal, and it removes large-shape global atomic contention.

## Profiling / cannsim interpretation

Sub-kernel cannsim may show the overflow-safe path has higher `SCALAR`/`PUSHQ` and wall cycles than the direct atomic path because `grid=1` cannot show FFTS/coreDim legality benefits. Document this honestly: compare traces for bottlenecks, but use hardware verification for the full target dispatch decision.

## Profiler guard

For read-only comparison baselines, guard the exact direct-grid formula before launch and return/print `SKIP coreDim_guard` or benchmark `inf` instead of poisoning the NPU context.
