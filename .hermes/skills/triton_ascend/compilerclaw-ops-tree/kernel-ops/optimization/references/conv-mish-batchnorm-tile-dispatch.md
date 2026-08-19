# Conv2d + Mish + BatchNorm: large-tile activation epilogue dispatch

## Pattern

For `Conv2d -> Mish (x * tanh(softplus(x))) -> BatchNorm2d`, keep Conv2d and BatchNorm on PyTorch/ACL and optimize only the non-standard Mish activation epilogue in Triton. This avoids replacing highly optimized ACL convolution/normalization while still removing the slow standard-op Mish chain.

## Optimization rule

Use a direct elementwise kernel for normal shapes and a separate persistent kernel only when the activation tile count exceeds the Ascend FFTS grid cap:

```python
_MAX_PROGRAMS = 65535
_BLOCK_SIZE = 8192  # large default activation tensors; benchmark against 4096

n_tiles = triton.cdiv(n_elements, _BLOCK_SIZE)
if n_tiles > _MAX_PROGRAMS:
    _mish_persistent_kernel[(_MAX_PROGRAMS,)](x, y, n_elements, _MAX_PROGRAMS, ...)
else:
    _mish_direct_kernel[(n_tiles,)](x, y, n_elements, ...)
```

The persistent loop must iterate over **tiles**, not elements:

```python
n_tiles = tl.cdiv(n_elements, BLOCK_SIZE)
for tile_id in range(pid, n_tiles, n_programs):
    offs = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
```

## Mish formula

Prefer the one-exp stable identity already used in several KernelBench kernels:

```python
use_large = x > 20.0
t = tl.exp(tl.where(use_large, 0.0, x))
den = t * t + 2.0 * t + 2.0
tanh_sp = tl.where(use_large, 1.0, 1.0 - 2.0 / den)
y = (x * tanh_sp).to(x_in.dtype)
```

This avoids unsupported/expensive `tl.tanh`/`tl.log` paths and matches PyTorch `softplus(beta=1, threshold=20)` closely when math is performed in fp32.

## Cannsim comparison when BLOCK changes

When increasing `BLOCK_SIZE`, compare **normalized cycles per element/tile-equivalent**, not raw wall cycles per program. A larger tile can have higher program wall cycles but still be faster end-to-end by reducing launch/program count. Report both:

```text
normalized_cycles_per_4096 = wall_cycles * 4096 / BLOCK_SIZE
hardware_time_ns = cycles * 0.4
```

## Profiling requirements

- Include tiny/irregular direct shapes and the default benchmark shape.
- Add a unit-only forced-persistent test by temporarily lowering `_MAX_PROGRAMS`; this validates the persistent dispatch path without allocating a massive tensor or risking grid-cap poisoning.
- Expect tiny shapes to regress slightly with larger tiles; optimize for the stated/default regime and document the trade-off.
