# Conv + Mish + Mish: ACL production dispatch with Triton fallback

Use this reference for KernelBench-style `Conv*d -> Mish -> Mish` pipelines where convolution is already handled by `nn.Conv*`/ACL and the editable Triton work is the post-convolution elementwise epilogue.

## Recognition

- The model keeps `self.conv = nn.Conv*d(...)` and applies two Mish activations to the convolution output.
- A Triton baseline may implement Mish manually with `softplus` + `tanh`, or may call `tl.tanh` directly.
- On Triton-Ascend, `tl.tanh` may be unavailable; `triton.language.math.tanh` can be used in diagnostic compile shims, but production optimized code should avoid relying on an invalid editable baseline.

## Optimization pattern

Keep convolution on ACL. For production, benchmark the native ACL activation chain before committing to a custom Triton epilogue:

```python
import torch.nn.functional as F

def mish_mish_triton(x):
    if _USE_ACL_DISPATCH:
        return F.mish(F.mish(x))
    return _mish_mish_triton_impl(x)
```

Retain a custom Triton fallback for direct/persistent dispatch coverage:

```python
@triton.jit
def _tanh_softplus_stable(x_f32):
    z = tl.exp(-tl.abs(x_f32))
    z2 = z * z
    pos = (1.0 + 2.0 * z) / (1.0 + 2.0 * z + 2.0 * z2)
    neg = (z2 + 2.0 * z) / (z2 + 2.0 * z + 2.0)
    return tl.where(x_f32 >= 0.0, pos, neg)

@triton.jit
def _mish_mish_direct_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offs = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0, care_padding=False).to(tl.float32)
    mish1 = x * _tanh_softplus_stable(x)
    out = mish1 * _tanh_softplus_stable(mish1)
    tl.store(y_ptr + offs, out, mask=mask)
```

## Cannsim methodology

If the editable baseline cannot compile because it uses missing `tl.tanh`, do not edit the baseline. For trace comparison only, create a clearly-labeled baseline-equivalent cannsim shim that uses `from triton.language.math import tanh as tl_tanh` to compile the intended math. Document this distinction in `performance_report.md`.

Compare custom Triton fallback traces with sub-kernel hosts (`grid=(1,1,1)`, one flattened activation tile). Expect reductions in `RVECEX`, `PUSHQ`, and vector instruction count from replacing log/tanh-heavy formulas with the one-exp stable ratio.

## Profiling and tests

- Include all providers: PyTorch/ACL, editable baseline, read-only `base_*.py`, optimized.
- If optimized production dispatch uses ACL, unit-test the custom Triton fallback by temporarily setting `_USE_ACL_DISPATCH = False`.
- Force direct and persistent fallback paths in unit tests: use a normal high grid cap for direct, then temporarily lower `_MAX_GRID` (for example to `1`) to exercise the persistent tile loop without huge tensors.
- Keep the production benchmark numbers honest: if ACL beats the custom Triton epilogue end-to-end, use ACL production dispatch and state that cannsim improvements apply to the fallback kernel, not the selected production path.
