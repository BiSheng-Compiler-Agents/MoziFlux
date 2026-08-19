# Optimizations Applied

## 1. Replace row/block epilogue with flat contiguous tiling

**Before:** the baseline epilogue launches one program for every `(row, column-block)` tile and uses `BLOCK_N` up to 1024.

```python
grid = (rows, triton.cdiv(cols, BLOCK_N))
_scale_hardtanh_gelu_kernel[grid](..., BLOCK_N=BLOCK_N, num_stages=1)
```

**After:** the optimized epilogue treats the contiguous `Linear` output as a flat vector and processes 4096 elements per program.

```python
n_elements = y.numel()
n_tiles = triton.cdiv(n_elements, _BLOCK_SIZE)
_scale_hardtanh_gelu_flat_kernel[(n_tiles,)](..., BLOCK_SIZE=4096, num_stages=2)
```

Rationale: `nn.Linear` returns contiguous output, so a flat tile removes per-row stride arithmetic and cuts the target full-shape epilogue program count from `2048 * ceil(8192/1024) = 16384` to `ceil(16777216/4096) = 4096`.

## 2. Keep a grid-capped persistent path for oversized outputs

```python
if n_tiles > _MAX_PROGRAMS:
    _scale_hardtanh_gelu_persistent_kernel[(_MAX_PROGRAMS,)](
        y, y, n_elements, _MAX_PROGRAMS, ..., BLOCK_SIZE=_BLOCK_SIZE
    )
```

Rationale: Ascend FFTS launch dimensions are capped at 65535 programs. The normal benchmark shape stays on the faster direct path, while the persistent path preserves correctness for larger accepted tensors and is covered by the forced-persistent unit test in `profile_kernels.py`.

## 3. Remove unsupported cache modifier and unsafe launch metadata

```python
x = tl.load(x_ptr + offsets, mask=mask, other=0.0, care_padding=False)
...
num_stages=2
```

Rationale: `cache_modifier=".cg"` is unsafe on Triton-Ascend, and `num_stages=1` is a known crash hazard. The optimized path uses masked contiguous loads with `care_padding=False` because padded lanes are not stored and do not affect reductions.

## 4. Preserve exact math and dtype behavior

```python
xf = x.to(tl.float32) * scale
xf = tl.minimum(tl.maximum(xf, minv), maxv)
y32 = 0.5 * xf * (1.0 + tl.math.erf(xf * 0.7071067811865476))
tl.store(y_ptr + offsets, y32.to(x.dtype), mask=mask)
```

Rationale: the kernel keeps PyTorch `nn.GELU(approximate="none")` semantics by using `erf`, and casts back to the original GEMM output dtype.
