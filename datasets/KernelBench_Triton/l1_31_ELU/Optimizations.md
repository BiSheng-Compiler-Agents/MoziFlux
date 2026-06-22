# Optimizations Applied

## 1. Legal large-tensor dispatch under Ascend FFTS grid cap

The baseline chooses direct one-program-per-tile dispatch. For the original shape `(4096, 393216)`, even with `BLOCK_SIZE=8192` it requires `196608` programs, exceeding Ascend's `coreDim <= 65535` limit.

```python
n_tiles = triton.cdiv(n_elements, _BLOCK_SIZE)
if n_tiles > _MAX_PROGRAMS:
    _elu_persistent_kernel[(_MAX_PROGRAMS,)](
        x_flat, y_flat, n_elements, self.alpha, _MAX_PROGRAMS,
        BLOCK_SIZE=_BLOCK_SIZE, num_warps=4, num_stages=2,
    )
else:
    _elu_direct_kernel[(n_tiles,)](...)
```

Rationale: keep the fast direct path for normal tensors while using a capped persistent path only when direct launch would overflow the FFTS grid.

## 2. Persistent loop iterates over tiles, not elements

```python
n_tiles = tl.cdiv(n_elements, BLOCK_SIZE)
for tile_id in range(pid, n_tiles, n_programs):
    ...
```

Rationale: grid-striding by tile id gives complete coverage without launching more than 65535 programs. Striding by element count would skip tiles and produce incorrect output.

## 3. 64-bit offsets in persistent path

The original tensor has ~1.61B fp32 elements (~6.44GB). The persistent path uses 64-bit element offsets to avoid byte-offset overflow on late tiles.

```python
base = tile_id.to(tl.int64) * BLOCK_SIZE
offsets = base + tl.arange(0, BLOCK_SIZE).to(tl.int64)
mask = offsets < n_elements
```

Rationale: an initial persistent implementation with 32-bit offsets failed correctness on the original shape; the 64-bit offset version passes full-shape hardware verification.

## 4. Contiguity/alignment hints and masked memory ops

```python
tl.multiple_of(offsets, 16)
tl.max_contiguous(offsets, BLOCK_SIZE)
x = tl.load(x_ptr + offsets, mask=mask, other=0.0, eviction_policy="evict_first")
tl.store(y_ptr + offsets, y, mask=mask)
```

Rationale: contiguous masked GM access is the correct vector-core pattern for elementwise activation kernels and preserves boundary safety for irregular shapes.

## 5. Profile covers all dispatch paths

`profile_kernels.py` tests direct, irregular direct, and original oversized dispatch:

```python
_BENCH_SHAPES = [
    ("direct_1M", (1024, 1024)),
    ("direct_irregular", (257, 4097)),
    ("persistent_original", (4096, 393216)),
]
```

The checked-in profiler uses `(4096, 393216)` for the original persistent case and pre-skips the editable baseline before an invalid `coreDim > 65535` launch.
