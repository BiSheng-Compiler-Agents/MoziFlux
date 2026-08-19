# Conv + Mish + Tanh epilogue optimization

Use this reference for KernelBench-style `Conv1d/2d/3d -> Mish -> Tanh` pipelines where convolution is already handled by `nn.Conv*`/ACL and the Triton work is the post-convolution elementwise epilogue.

## Recognition

- The baseline keeps `self.conv = nn.Conv*d(...)` and launches a Triton elementwise activation over the convolution output.
- The epilogue computes `tanh(mish(x)) = tanh(x * tanh(softplus(x)))` with manual exp-based formulas.
- Cannsim trace shows high `RVECEX`/`PUSHQ` from multiple `RV_VEXP`, `RV_VMUL*`, `RV_VADD*`, and `RV_VDIV` groups; MTE load/store lanes are mostly fixed by the one input tile and one output tile.

## Optimization pattern

Keep the vendor convolution path and optimize only the epilogue:

```python
from triton.language.math import tanh as tl_tanh

@triton.jit
def _tanh_softplus_stable(x_f32):
    # z = exp(-abs(x)); avoids exp(2*x) overflow and removes one exp group.
    z = tl.exp(-tl.abs(x_f32))
    z2 = z * z
    pos = (1.0 + 2.0 * z) / (1.0 + 2.0 * z + 2.0 * z2)
    neg = (z2 + 2.0 * z) / (z2 + 2.0 * z + 2.0)
    return tl.where(x_f32 >= 0.0, pos, neg)

@triton.jit
def _mish_tanh_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    tl.max_contiguous(offs, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0, care_padding=False)
    x_f32 = x.to(tl.float32)
    mish = x_f32 * _tanh_softplus_stable(x_f32)
    y = tl_tanh(mish).to(x.dtype)
    tl.store(y_ptr + offs, y, mask=mask)
```

Rationale:
- Replace `exp(2 * max(x, 0)) * (1 + exp(-abs(x)))**2` with a one-exp stable ratio for `tanh(softplus(x))`.
- Replace manual final `tanh` (`abs`, `exp`, sign, division) with `tl.math.tanh`.
- Keep contiguous 1D tiles and `care_padding=False`; the operation is pure elementwise and masked tail values do not feed reductions.

## Dispatch pattern

Use direct dispatch for normal sizes and a separate persistent fallback only above Ascend's 65,535 grid cap:

```python
n_tiles = triton.cdiv(n_elements, BLOCK_SIZE)
if n_tiles > _MAX_GRID:
    _persistent_kernel[(_MAX_GRID,)](x, y, n_elements, _MAX_GRID, BLOCK_SIZE=BLOCK_SIZE)
else:
    _direct_kernel[(n_tiles,)](x, y, n_elements, BLOCK_SIZE=BLOCK_SIZE)
```

The persistent kernel must iterate over tiles:

```python
n_tiles = tl.cdiv(n_elements, BLOCK_SIZE)
for tile_id in tl.range(pid, n_tiles, n_programs, num_stages=2):
    ...
```

## Verification notes

- Unit-test both direct and persistent paths. For persistent, temporarily lower the module's grid cap (for example `_MAX_GRID = 1`) so a modest tensor exercises the persistent loop without huge allocation.
- Cannsim sub-kernel should trace only the epilogue (`grid=(1,1,1)`, one `BLOCK_SIZE` tile). Full convolution is not the target of this diagnostic.
- Expect `RVECEX` and `PUSHQ` reductions; a rise in `RV_VDIV` can still be a net win if exp/mul/add groups shrink more.
- End-to-end PyTorch/ACL can still beat the Triton epilogue when convolution dominates. Report that honestly; the Triton optimization is judged against the editable Triton baseline unless the task asks for pure ACL dispatch.
