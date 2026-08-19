# ConvTranspose3d + Swish + GroupNorm + HardSwish ACL Dispatch

Use this reference when optimizing KernelBench-style models that run `nn.ConvTranspose3d` followed by Swish, GroupNorm, and HardSwish, especially when the editable baseline implements the post chain with custom Triton reduction/apply kernels.

## Recognition pattern

- Convolution is already `nn.ConvTranspose3d` / ACL.
- Baseline custom Triton post-processing computes Swish statistics with atomics or partial sums, then reloads the convolution output and recomputes Swish for GroupNorm + HardSwish.
- Default 3D shapes create very large launch grids or slow/toxic comparison runs.
- Cannsim microprobes show scalar-heavy index decode, `SCALAR`/`SCALARLDST`, `PUSHQ`, MTE waits, or atomics rather than useful Cube work.

## Preferred production optimization

Keep the convolution on ACL and dispatch the standard post chain to native PyTorch/NPU operators instead of defaulting to custom Triton:

```python
y = self.conv_transpose(x)
y = y * torch.sigmoid(y)  # Swish

y = torch.nn.functional.group_norm(
    y,
    self.group_norm.num_groups,
    self.group_norm.weight,
    self.group_norm.bias,
    self.group_norm.eps,
)

# Prefer algebraic HardSwish over F.hardswish on Ascend stacks where the native op may fail.
y = y * torch.clamp(y + 3.0, min=0.0, max=6.0) * (1.0 / 6.0)
return y
```

Rationale: the ACL path avoids custom atomic reductions, repeated Swish computation, scalar-heavy index decode, and grid-cap/context-poisoning risk. On small/medium shapes it can be ~1.5-2x faster than the custom Triton baseline, while default-shape baseline timing may need to be preskipped if the comparison launch is toxic.

## Triton fallback requirements

If retaining a custom Triton fallback:

- Keep it disabled by default unless hardware results prove it wins.
- Prefer per-group partial reductions over atomic accumulation where possible.
- Add both direct and persistent apply paths, gated by `cdiv(n_elements, BLOCK_SIZE) > 65535`.
- Unit-test hidden fallback paths by temporarily forcing the Triton route and lowering the grid cap to exercise persistent mode on a modest tensor.

## Profiling/reporting notes

- Keep `Baseline Triton2` parser-visible even when `base_*.py` is read-prohibited by the sandbox: emit `TEST Baseline Triton2 <label>: SKIP_UNAVAILABLE ... max_abs=inf` and `inf` benchmark cells.
- If `torch._C._nn.hardswish`/`F.hardswish` fails on the target Ascend stack, use the algebraic formula above for both reference and optimized paths; do not record a durable claim that HardSwish is universally unavailable.
- Cannsim can compare tiny fallback microkernels, but it does not measure the production benefit of routing the standard post chain to ACL. Use remote hardware latency for the production decision.
