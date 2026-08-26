# Row-wise Cosine Similarity Loss Pattern

Use this for `mean(1 - cosine_similarity(x, y, dim=1))`-style kernels where each row is an independent dot/norm reduction and the final output is a scalar mean.

## Recognition signals

- Inputs are 2D `(B, D)` tensors with equal shape.
- The custom Triton baseline launches one program per row, computes `dot`, `sum(x*x)`, `sum(y*y)`, stores a per-row loss vector, then returns `out.mean()` from PyTorch.
- The baseline already makes inputs contiguous on the host, but the kernel still carries `stride_xb/stride_xd/stride_yb/stride_yd` runtime arguments.

## Recommended first optimization

Keep the same row-kernel structure and reduce address-generation overhead by relying on the host-side contiguous copy:

```python
x = predictions.contiguous()
y = targets.contiguous()
base = row * D + tl.arange(0, BLOCK_SIZE)
xv = tl.load(x_ptr + base, mask=mask, other=0.0).to(tl.float32)
yv = tl.load(y_ptr + base, mask=mask, other=0.0).to(tl.float32)
```

Preserve fp32 reductions:

```python
dot = tl.sum(xv * yv, axis=0)
nx2 = tl.sum(xv * xv, axis=0)
ny2 = tl.sum(yv * yv, axis=0)
loss = 1.0 - dot / tl.maximum(tl.sqrt(nx2) * tl.sqrt(ny2), EPS)
```

## Grid-cap legality path

The direct row grid is fastest for normal shapes but illegal when `B > 65535`. Add a persistent row-loop only above that threshold:

```python
if B <= 65535:
    _direct[(B,)](...)
else:
    _persistent[(65535,)](..., n_programs=65535)
```

Inside the persistent kernel, loop over rows, not elements or D-tiles:

```python
row = tl.program_id(0)
while row < B:
    ...
    row += n_programs
```

## What not to assume

- Do not automatically dispatch to ACL/PyTorch just because this is a loss-like operation. For row-wise cosine similarity at moderate `D`, a custom Triton row kernel can beat ACL; measure before replacing it.
- A fused scalar-output atomic reduction can regress from scalar/PUSHQ overhead. Keep the per-row loss + `mean()` structure unless a measured two-stage reducer wins on hardware.
- A single-sqrt rewrite (`1 / sqrt(nx2 * ny2)`) is not guaranteed faster; verify with cannsim/hardware before adopting.

## Reporting notes

- Compare cannsim direct row kernels at the same `B=1, D=<row width>` sub-kernel so changes isolate instruction mix.
- Hardware verification should include a synthetic persistent-dispatch correctness shape (`B=65536, D=1` or similar) that is small in memory but crosses the row-grid cap.
