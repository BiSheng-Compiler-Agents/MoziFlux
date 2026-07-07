# Algebraic dead-GEMM zero-output pattern

## When it applies

Use this pattern when the model computes an expensive projection/reduction only to subtract a tensor from itself before a zero-preserving activation, for example:

```python
y = linear(x)
m = torch.max(y, dim=1, keepdim=True).values
out = F.gelu(m - m)   # exactly zero
```

For the active dispatch contract, `m - m` is exactly zero and `GELU(0) == 0`, so the GEMM, reduction, subtraction, and activation are dead work.

## Implementation pattern

Emit the required zero-shaped output directly, but keep the original host interface and a fallback for non-hot constructor arguments:

```python
if self.max_dim == 1:
    out = torch.empty((x.shape[0], 1), device=x.device, dtype=x.dtype)
    n_tiles = triton.cdiv(out.numel(), BLOCK)
    if n_tiles > MAX_PROGRAMS:
        zero_persistent[(MAX_PROGRAMS,)](out, out.numel(), MAX_PROGRAMS, BLOCK=BLOCK)
    else:
        zero_direct[(n_tiles,)](out, out.numel(), BLOCK=BLOCK)
    return out

# Fallback preserves the broader interface.
y = self.gemm(x)
m = torch.max(y, dim=self.max_dim, keepdim=True).values
return F.gelu(m - m)
```

Direct zero-fill kernel:

```python
@triton.jit
def zero_direct(out, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.multiple_of(offs, 16)
    tl.max_contiguous(offs, BLOCK)
    tl.store(out + offs, tl.zeros((BLOCK,), tl.float32), mask=mask)
```

## Profiling and verification notes

- Cannsim sub-kernel traces compare only the zero-fill kernel; the end-to-end win comes from algebraic elimination of dead GEMM work.
- Keep both direct and persistent zero-fill dispatch paths, and force-test persistent by temporarily lowering `MAX_PROGRAMS` in `profile_kernels.py`.
- If the sandbox forbids reading `base_*.py`, keep the Baseline Triton2 column parser-visible with `SKIP_REFERENCE_SANDBOX`/`inf` rather than importing the reference file.

## Pitfalls

- Do not apply if the subtract operands can differ, if NaN propagation semantics matter for the task, or if the baseline intentionally relies on side effects from the dead-looking operation.
- Do not add a new runtime restriction; preserve the original constructor/interface with a fallback path.
