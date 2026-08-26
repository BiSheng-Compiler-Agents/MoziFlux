# ReLU + HardSwish algebraic epilogue

## Pattern

For an epilogue that computes `HardSwish(ReLU(x))`, avoid materializing the ReLU value with a separate vector max when the output can be expressed directly as a piecewise formula:

```python
# Baseline
r = tl.maximum(x, 0.0)
y = r * tl.minimum(r + 3.0, 6.0) * (1.0 / 6.0)

# Optimized
y_pos = x * tl.minimum(x + 3.0, 6.0) * (1.0 / 6.0)
y = tl.where(x > 0.0, y_pos, 0.0)
```

This is exactly equivalent because `ReLU(x)=0` for `x <= 0`, and for `x > 0` the original expression becomes `x * min(x + 3, 6) / 6`.

## When to use

- Post-convolution or post-linear contiguous activation epilogues where the intermediate ReLU result is not reused.
- Direct elementwise Triton kernels with masked contiguous loads/stores.
- Combine with direct + persistent dispatch for large tensors that may exceed Ascend's FFTS grid cap.

## Expected cannsim signal

The simplification can remove `RV_VMAXS` work and lower `RVECEX`, `PUSHQ`, and vector-tail wait time. In one fp32 activation tile it reduced wall cycles by about 11% and trace event count by about 25%.

## Pitfalls

- Do not use this rewrite if the post-ReLU tensor is needed elsewhere; it assumes only the final HardSwish output is required.
- Small tensors may regress from the alternate expression despite improving larger/default shapes. If small-shape latency matters, benchmark a small-shape fallback before adopting globally.
- Keep exact piecewise semantics; do not approximate HardSwish unless the task explicitly permits it.
