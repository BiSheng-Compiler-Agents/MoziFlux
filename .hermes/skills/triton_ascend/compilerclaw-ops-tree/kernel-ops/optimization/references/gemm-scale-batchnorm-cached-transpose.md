# GEMM + Scale + BatchNorm: Cached Weight Transpose Pattern

Use this reference when optimizing `nn.Linear`-style GEMM epilogues on Ascend where the editable Triton baseline reads `weight` in PyTorch's native `[N, K]` layout but the kernel consumes it as logical `[K, N]`.

## Symptom

A baseline GEMM kernel may load B with non-contiguous column access:

```python
B = self.gemm.weight.contiguous()  # [N, K]
b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
# stride_bn == K, so adjacent N columns are strided by K
```

This often shows as high MTE/scalar/vector-side overhead and poor scaling on large-K linear layers.

## Pattern

Cache a host-side contiguous transpose and make the kernel consume B as `[K, N]`:

```python
class ModelNew(nn.Module):
    def __init__(self, ...):
        ...
        self._cached_weight_kn = None
        self._cached_weight_key = None

    def _weight_kn(self):
        w = self.gemm.weight
        key = (w.data_ptr(), tuple(w.shape), w.dtype, w.device, getattr(w, "_version", 0))
        if self._cached_weight_key != key or self._cached_weight_kn is None:
            self._cached_weight_kn = w.transpose(0, 1).contiguous()
            self._cached_weight_key = key
        return self._cached_weight_kn
```

Then launch the kernel with `B = self._weight_kn()` and `stride_bn == 1` so the N dimension is contiguous.

## Combine With

- Larger Cube tile if UB/L1 budget allows, e.g. 128×128×32 for fp32 GEMM.
- 1D grouped scheduling (`GROUP_M`) over logical M/N tiles to improve B-tile reuse.
- `tl.range(0, K, BLOCK_K)` instead of pointer-advance `while` loops.
- In-place Cube accumulation: `tl.dot(a, b, acc)`.
- `al.compile_hint(acc, "dot_pad_only_k")` when M/N blocks are already 16-aligned.
- A correctness-preserving hybrid dispatch when the fully fused Triton GEMM is compile-unstable or slower at the required block-aligned production shape: use ACL/PyTorch `linear(...).contiguous()` for dense GEMM, then apply the cached eval-BN affine (`y * alpha + beta2`), while keeping a tested Triton fused path for the irregular/custom dispatch regime.

## Correctness Notes

- Keep BatchNorm separate unless training/eval running-stat semantics are explicitly replicated.
- Cache invalidation should include at least data pointer, shape, dtype, device, and parameter version.
- If the model migrates device/dtype, clear the cached transpose.

## Observed Outcome

On a GEMM+scale+BatchNorm workload with required shape 1024×8192 × 8192×8192, combining cached `[K,N]` transpose with 128×128×32 tiles, grouped 1D scheduling, `tl.range`, `tl.dot(a,b,acc)`, and `dot_pad_only_k` reduced hardware latency from ~733 ms to ~48.6 ms vs the editable Triton baseline, while preserving optimized correctness across small/medium/required shapes.
