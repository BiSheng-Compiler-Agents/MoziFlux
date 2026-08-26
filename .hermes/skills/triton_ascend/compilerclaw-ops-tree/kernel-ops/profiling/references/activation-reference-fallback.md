# Activation reference fallback

## Pattern

When `profile_kernels.py` builds the PyTorch/ACL reference, a native activation may fail on a specific CANN/torch_npu build even though the optimized kernel and the mathematical operation are valid. Do not let a failing reference provider make optimized correctness untestable. Replace only the reference expression with an exact algebraic equivalent, then keep every optimized/provider comparison gated against that reference.

## Example: ReLU(HardSwish)

Native form:

```python
return F.relu(F.hardswish(x))
```

Exact fallback:

```python
y_pos = x * torch.clamp(x + 3.0, max=6.0) * (1.0 / 6.0)
return torch.where(x > 0.0, y_pos, torch.zeros_like(x))
```

Why this is exact: for `x <= 0`, the final ReLU makes the result zero; for `x > 0`, `HardSwish(x) = x * min(x + 3, 6) / 6`.

## Requirements

- Document the fallback in `profile_kernels.py` so the reference is auditable.
- Do not hide optimized failures: optimized correctness must still compare against this reference on every benchmark shape.
- Keep comparison-provider failures parser-visible with `SKIP_UNAVAILABLE` only when the provider is read-only or outside the deliverable scope.
