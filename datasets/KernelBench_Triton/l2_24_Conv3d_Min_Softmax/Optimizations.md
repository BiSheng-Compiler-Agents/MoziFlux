# Optimizations Applied

## 1. Hybrid dispatch: Triton for small/medium post-op, ACL for large post-op

Hardware verification showed the original fused Triton epilogue is best for small/medium shapes, while the target shape is faster with mature ACL reductions. The optimized host keeps the fused Triton path below the measured threshold and routes larger post-conv tiles to `torch.amin + softmax`.

```python
_ACL_TILE_THRESHOLD = 1024
x = self.conv(x).contiguous()
_, C, _, H, W = x.shape
total_tiles = x.shape[0] * H * triton.cdiv(W, _BLOCK_W)
if _next_pow2(C) <= 64 and total_tiles < _ACL_TILE_THRESHOLD:
    return _fused_triton(x)
return F.softmax(torch.amin(x, dim=2), dim=1)
```

Rationale: target hardware latency improved from Baseline Triton1 `1.771558 ms` to optimized `1.439981 ms`; small/medium remain on the fast fused Triton path (`0.013277 ms`, `0.088387 ms`).

## 2. Grid-capped persistent fused fallback

For valid large `(B,H,W)` products where Triton remains selected, the optimized file includes a persistent fused kernel capped at Ascend's 65,535 program limit.

```python
if total_tiles > _MAX_PROGRAMS:
    _fused_minD_softmaxC_persistent[(_MAX_PROGRAMS,)](..., total_tiles, _MAX_PROGRAMS)
else:
    _fused_minD_softmaxC_direct[(total_tiles,)](...)
```

Inside the persistent kernel, work advances by tile count:

```python
tile = tl.program_id(axis=0)
while tile < total_tiles:
    ...
    tile += n_programs
```

## 3. Preserve numerically stable fused min + softmax

The retained Triton path keeps the original single-kernel fusion, fp32 reduction values, masked min padding with `+inf`, and max-subtracted softmax.

```python
vals = tl.load(ptrs, mask=mask, other=float("inf"), care_padding=False).to(tl.float32)
run_min = tl.minimum(run_min, tl.min(vals, axis=1))
x_max = tl.max(tl.where(c_mask[:, None], run_min, -float("inf")), axis=0)
exps = tl.exp(run_min - x_max[None, :])
```

## 4. Correctness fallback for wider channel counts

The fused Triton implementation covers `C <= 64`, including the benchmark `C=24`. Wider channel counts route to ACL, and `profile_kernels.py` includes `fallback_c80` to test this dispatch path.
