# Optimizations

## 1. Grid-cap dispatch for target shape

Baseline launches one Triton program per output element:
```python
grid = (N * C * OH * OW,)
```
For the target shape this is `16*64*186*186 = 35,436,544` programs, above Ascend's `coreDim <= 65535` launch cap. The optimized host keeps the direct path for small shapes and routes large shapes to a persistent kernel:
```python
if total_tiles <= _MAX_PROGRAMS:
    _avg_pool2d_tile_kernel[(total_tiles,)](...)
else:
    _avg_pool2d_persistent_kernel[(_MAX_PROGRAMS,)](..., total_tiles, _MAX_PROGRAMS, ...)
```

## 2. No-padding fast path

`get_init_inputs()` uses `kernel_size=11` with default `padding=0`, so every pooling window is in-bounds by the output-size formula. The optimized kernel removes per-element boundary branches and dynamic `count` updates:
```python
for kh in tl.static_range(0, KH):
    row_ptr = x_base + (ih0 + kh) * in_stride_h
    for kw in tl.static_range(0, KW):
        acc += tl.load(row_ptr + (iw0 + kw) * in_stride_w).to(tl.float32)
tl.store(y_ptr + y_off, acc / ((KH * KW) + 0.0))
```
For nonzero padding the host falls back to `torch.nn.functional.avg_pool2d(..., count_include_pad=False)` to preserve baseline semantics instead of launching an incorrect fast path.

## 3. Compile-safe scalar pooling body

A vectorized output-width tile was attempted but BiSheng exceeded VF stack size (`17952 > 6144`). The final optimized kernel keeps one output per program and focuses on launch legality plus scalar-branch reduction, which cannsim confirms compiles and runs.
