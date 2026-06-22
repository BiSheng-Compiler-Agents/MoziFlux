# Optimizations Applied

## 1. Overflow-safe persistent dispatch for target tensor

The input tensor has `112*64*512*512 = 1,879,048,192` elements. The source kernel uses `BLOCK=16384`, producing `114,688` programs for the reduction and scale launches, which exceeds Ascend FFTS `coreDim <= 65535`.

```python
n_tiles = triton.cdiv(n_elements, _BLOCK_ELEMS)
n_programs = min(n_tiles, _MAX_PROGRAMS)
...
_partial_sumsq_kernel[(n_programs,)](...)
_scale_kernel[(n_programs,)](...)
```

Rationale: capping the launch grid avoids target-shape runtime failure while preserving coverage by looping over tiles inside each program.

## 2. Replace full-shape global atomic serialization with partial reductions

For oversized tensors, each program writes one partial sum instead of contending on one global scalar with `tl.atomic_add`.

```python
ss = tl.sum(x * x, axis=0)
tl.store(partials_ptr + tile, ss)
...
vals = tl.load(partials_ptr + idx, mask=mask, other=0.0, care_padding=False)
acc += tl.sum(vals, axis=0)
```

Rationale: partial sums remove multi-core atomic contention and make the large target path deterministic; a single final reducer only scans the compact partials array.

## 3. Two dispatch paths: direct small path, overflow-safe large path

The optimized host keeps the two-launch atomic path when the grid is legal and routes only grid-overflow cases to the three-launch partial path.

```python
if n_tiles <= _MAX_PROGRAMS:
    sumsq = torch.zeros((1,), device=x_contig.device, dtype=torch.float32)
    _sumsq_atomic_kernel[(n_tiles,)](x_contig, n_elements, sumsq, BLOCK=_BLOCK_ELEMS)
else:
    partials = torch.empty((n_tiles,), device=x_contig.device, dtype=torch.float32)
    _partial_sumsq_kernel[(n_programs,)](...)
    _reduce_partials_kernel[(1,)](...)
```

Rationale: direct dispatch avoids extra launch overhead on small/medium shapes; the persistent partial path is reserved for the benchmark target where the baseline is invalid.

## 4. Contiguous, masked fp32 reduction and fp32 output

All element accesses are contiguous, boundary masked, and reductions upcast to fp32.

```python
idx = tile * BLOCK + tl.arange(0, BLOCK)
mask = idx < n_elements
x = tl.load(x_ptr + idx, mask=mask, other=0.0, care_padding=False).to(tl.float32)
y = x * tl.rsqrt(tl.load(sumsq_ptr))
tl.store(y_ptr + idx, y, mask=mask)
```

Rationale: contiguous access maximizes GM→UB transfer efficiency; fp32 accumulation matches the PyTorch reference and source kernel output promotion.
