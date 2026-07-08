# Grid-Cap ACL Fallback for Standard Epilogues

Use this pattern when a custom Triton epilogue is correct and faster on small/medium shapes, but the default/full benchmark shape would exceed Ascend's `coreDim <= 65535` launch cap or a persistent Triton rewrite is slower than the standard library path.

## Recognition pattern

- The optimized Triton path is an elementwise/reduction epilogue around a standard operator output, e.g. LayerNorm + GELU + scale after convolution.
- The natural tile count for the full/default shape exceeds `65535` even after reasonable row blocking.
- A persistent-grid Triton version is valid but MTE/scalar-heavy, or remote profiling shows it much slower than the ACL/PyTorch standard library implementation.
- The large-shape operation is covered by `torch.nn.functional` / ACL (e.g. `layer_norm`, `gelu`, pooling, convolution).

## Dispatch pattern

```python
_MAX_PROGRAMS = 65535
_BLOCK_ROWS = 16

n_tiles = triton.cdiv(total_rows, _BLOCK_ROWS)
if int(n_tiles) > _MAX_PROGRAMS:
    # ACL-backed fallback for the grid-cap regime.
    y = x.permute(0, 2, 3, 4, 1).contiguous()
    y = F.layer_norm(y, (channels,), weight, bias, eps)
    y = F.gelu(y, approximate="none") * scale
    return y.permute(0, 4, 1, 2, 3).contiguous()

_kernel[(int(n_tiles),)](...)
```

Keep the optimized Triton path for shapes that fit under the cap; do not force a persistent Triton path just to avoid ACL if remote hardware shows ACL is faster and the operation is a standard library primitive.

## Profiling requirements

- Include one benchmark/unit-test shape below the cap and one exact/default shape above the cap, with labels such as `direct_small` and `gridcap_default`.
- Pre-skip comparison providers that would poison the NPU context with `coreDim > 65535`; report `inf` with neutral `INFO` wording.
- Document that cannsim traces cover only the direct Triton sub-kernel when the optimized full-shape dispatch uses ACL fallback; hardware latency must come from `remote_verify`.
