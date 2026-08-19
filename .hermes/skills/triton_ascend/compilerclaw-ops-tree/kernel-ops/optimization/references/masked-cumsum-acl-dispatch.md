# Masked cumsum: custom Triton scan vs ACL dispatch

## When to use

Use this when optimizing masked cumulative sum operators where the baseline first applies a boolean mask and then runs a custom Triton row-wise `tl.cumsum` scan.

## Recognition pattern

```python
y = x * mask.to(dtype=x.dtype)
# optional movedim/contiguous/view to flatten rows
_cumsum_lastdim_kernel[(m_size,)](...)
```

The custom kernel usually processes each row in chunks with `tl.cumsum(vals, axis=0)` plus a scalar carry between chunks. On Ascend this can remain `SCALARLDST`/scalar-control limited even when the kernel is otherwise vector-shaped.

## Recommended production path

If the operation is exactly masked cumsum, prefer dispatching the standard ACL/PyTorch scan:

```python
def masked_cumsum(x, mask, dim=-1):
    if x.device.type != "npu" or mask.device.type != "npu":
        raise ValueError("masked_cumsum requires NPU tensors")
    if x.shape != mask.shape:
        raise ValueError("x and mask must have the same shape")
    if x.ndim == 0:
        raise ValueError("masked_cumsum requires at least one dimension")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError("masked_cumsum supports float16, bfloat16, and float32 only")

    dim = dim % x.ndim
    return torch.cumsum(x * mask.to(dtype=x.dtype), dim=dim)
```

Preserve the baseline's validation and constructor semantics. Do not add shape-specific guards or benchmark-shape assumptions.

## Cannsim guidance

- Run cannsim on the original custom scan body as the baseline bottleneck probe.
- A fused custom diagnostic candidate (`load x`, `load mask`, zero masked lanes, `tl.cumsum`) is worth testing only as an A/B probe; if it remains `SCALARLDST`-bound or regresses, do not ship it as production just to keep a custom Triton launch.
- For the ACL production path, report custom Triton device-kernel cycles as `0` / "custom launch removed" and rely on `remote_verify` for hardware latency.

## Profiling guidance

Keep all provider columns visible: PyTorch / ACL, editable baseline, read-only `base_*.py`, and optimized. For huge row-scan target shapes, pre-skip comparison Triton providers with a neutral `compile_guard`/`inf` if compiling them would poison or time out verification; still require optimized PASS on every benchmark shape.
