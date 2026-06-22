# Optimizations for 30_Softsign

## 1. Larger vector tile: 1024 -> 8192 elements

Baseline used one program per 1024 elements:

```python
block_size = 1024
grid = (triton.cdiv(n_elements, block_size),)
```

Optimized direct path uses an 8192-element tile:

```python
_BLOCK_SIZE = 8192
n_tiles = triton.cdiv(n_elements, _BLOCK_SIZE)
_softsign_direct_kernel[(n_tiles,)](..., BLOCK_SIZE=_BLOCK_SIZE)
```

Rationale: Softsign is a memory/vector elementwise op (`abs + add + div`), so a larger tile amortizes scalar launch/setup overhead and reduces program count while staying within UB for a small number of live fp32 vectors.

## 2. Two-path direct/persistent dispatch for Ascend FFTS grid cap

The original shape has `4096 * 393216 = 1,610,612,736` elements. Baseline direct dispatch with block 1024 needs 1,572,864 programs, exceeding the Ascend `coreDim <= 65535` grid cap.

```python
if n_tiles > _MAX_PROGRAMS:
    _softsign_persistent_kernel[(_MAX_PROGRAMS,)](
        x_flat, y_flat, n_elements, _MAX_PROGRAMS, BLOCK_SIZE=_BLOCK_SIZE
    )
else:
    _softsign_direct_kernel[(n_tiles,)](
        x_flat, y_flat, n_elements, BLOCK_SIZE=_BLOCK_SIZE
    )
```

The persistent kernel iterates over tile ids, not elements:

```python
n_tiles = tl.cdiv(n_elements, BLOCK_SIZE)
for tile_id in range(pid, n_tiles, n_programs):
    offsets = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
```

Rationale: direct remains fastest for normal-size tensors; persistent only activates for oversized tensors to keep launch grid legal.

## 3. Contiguity/alignment compiler hints

```python
offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
tl.multiple_of(offsets, 16)
tl.max_contiguous(offsets, BLOCK_SIZE)
```

Rationale: all loads/stores are contiguous; hints help Triton-Ascend generate coalesced vector memory movement and avoid conservative address analysis.

## 4. Masked contiguous load/store and input-preserving dtype

```python
x = tl.load(x_ptr + offsets, mask=mask, other=0.0, eviction_policy="evict_first")
y = x / (tl.abs(x) + 1.0)
tl.store(y_ptr + offsets, y, mask=mask)
```

Rationale: masked OOB-safe memory access satisfies Ascend boundary requirements. Softsign has no reduction, so computing in input dtype avoids unnecessary explicit fp32 upcasts.
