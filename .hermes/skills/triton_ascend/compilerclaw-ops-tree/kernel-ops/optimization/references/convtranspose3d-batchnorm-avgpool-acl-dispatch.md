# ConvTranspose3d + BatchNorm3d + repeated AvgPool3d ACL dispatch

Use this reference for KernelBench-style models with `nn.ConvTranspose3d`, `nn.BatchNorm3d`, then two standard `AvgPool3d(kernel=2, stride=2)` calls, especially when the editable provider replaced the two pools with one custom Triton `AvgPool3d(kernel=4, stride=4)`.

## Recognition pattern

- The convolution and BatchNorm are already standard PyTorch/ACL modules.
- Custom Triton only handles the post-pooling stage.
- Baseline pooling launch maps one program per `(N*C*OD*OH, OW_tile)` row and can exceed Ascend's FFTS grid cap at default 3D volumes (example: `64*16*15*15 = 230400 > 65535`).
- Cannsim shows the custom pool micro-kernel is MTE2/VEC wait heavy; retiled row-block kernels improve normalized cycles/output but may still lose to ACL on hardware.

## Preferred production path

Use native ACL pooling for production when hardware verifies it is faster, and keep custom Triton only as a tested fallback:

```python
_USE_ACL_DISPATCH = True

if _USE_ACL_DISPATCH:
    return F.avg_pool3d(F.avg_pool3d(x, kernel_size=2, stride=2), kernel_size=2, stride=2)
```

This preserves the exact math and avoids grid-cap risk for standard post ops.

## Triton fallback pattern

Retile fused `k=4,s=4` pooling by multiple rows per CTA and cap the fallback grid:

```python
_ROWS_PER_CTA = 8
_BLOCK_W = 32
n_tiles = cdiv(total_rows, _ROWS_PER_CTA) * cdiv(OW, _BLOCK_W)
if n_tiles > _MAX_GRID:
    persistent[(65535,)](...)
else:
    direct[(n_tiles,)](...)
```

Unit tests must temporarily disable ACL dispatch and force both fallback paths:

```python
old_acl = opt_mod._USE_ACL_DISPATCH
old_grid = opt_mod._MAX_GRID
opt_mod._USE_ACL_DISPATCH = False
opt_mod._MAX_GRID = 1  # force persistent on a small tensor
# run correctness vs ACL reference
opt_mod._USE_ACL_DISPATCH = old_acl
opt_mod._MAX_GRID = old_grid
```

## Cannsim reporting

- Compare custom pool kernels by normalized `cycles/output`, not raw wall cycles, when row-blocking changes outputs per CTA.
- A wrapper-level `UNSAFE EARLY EXIT` can still leave a usable `instr.bin`; if manual `cannsim report` produces `trace_core0.json`, report the trace as recovered and note the wrapper failure.
- Do not claim cannsim measures the ACL production path; use `remote_verify` hardware latency for the production decision.

## Profiling requirements

- Keep `Baseline Triton2` parser-visible even if `base_*.py` is sandboxed/read-only: print `SKIP_UNAVAILABLE ... max_abs=inf` and `inf` benchmark cells.
- Pre-skip comparison providers that would overflow `coreDim > 65535` at the default shape; do not poison the NPU context before optimized timing.
- Include forced direct/persistent fallback unit tests in addition to production ACL correctness.
