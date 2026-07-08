# LeakyReLU branch-select epilogue pattern

## When to use

Use for pure pointwise epilogues that compute division/scaling followed by LeakyReLU on a contiguous tensor, especially after a standard ACL/PyTorch convolution or linear layer:

```python
y = conv_or_linear(x)
y = F.leaky_relu(y * scale, negative_slope=slope)
```

## Anti-pattern

A mathematically branchless LeakyReLU rewrite can emit extra vector min/add work on Ascend:

```python
y = x * inv
y_neg = tl.minimum(y, zero)
out = y + y_neg * (slope - one)
```

Typical cannsim symptoms:
- Many `RV_VMINS` events.
- High RVECEX/RVECLD/RVECST and PUSHQ relative to a simple pointwise operation.
- MTE3 remains the wall bottleneck, but vector-stream length inflates store-side wait.

## Pattern

Prefer compare/select form and keep the epilogue as a direct contiguous tile with a persistent fallback only above the FFTS grid cap:

```python
@triton.jit
def _leakyrelu_direct(x_ptr, y_ptr, n_elements, inv, slope, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0, care_padding=False)
    y = x * tl.full((), inv, x.dtype)
    s = tl.full((), slope, x.dtype)
    z = tl.full((), 0.0, x.dtype)
    out = tl.where(y >= z, y, y * s)
    tl.store(y_ptr + offs, out, mask=mask)
```

Host routing:

```python
n_tiles = triton.cdiv(n_elements, BLOCK_SIZE)
if n_tiles <= 65535:
    _leakyrelu_direct[(n_tiles,)](..., BLOCK_SIZE=BLOCK_SIZE)
else:
    _leakyrelu_persistent[(65535,)](..., n_programs=65535, BLOCK_SIZE=BLOCK_SIZE)
```

## Verification requirements

- Compare exact output against `F.leaky_relu(x * scale, negative_slope=slope)` or the full PyTorch/ACL chain.
- Unit-test the direct path and force-test the persistent path by temporarily lowering the grid cap.
- If `@perf_report`/`do_bench` reports impossible latencies for full model paths, switch that profile script to manual `time.perf_counter() + torch.npu.synchronize()` with low bounded reps and record this choice in the report.

## Observed evidence

In a Conv2d + divide + LeakyReLU epilogue probe with one contiguous 8192-element FP32 tile, branch-select changed cannsim wall cycles from 4287 to 3773 (1.136x), reduced x_events from 1665 to 871, and removed `RV_VMINS` from the top instruction list. The dominant pipe stayed MTE3, but RVECEX dropped from 802 to 275 busy cycles and PUSHQ from 848 to 321 busy cycles.
