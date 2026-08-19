# ConvTranspose3d + LayerNorm + AvgPool + GELU Hybrid Dispatch

Use this reference when a KernelBench-style operator runs a standard `nn.ConvTranspose3d`, then replaces standard post-processing (`LayerNorm`, `AvgPool3d`, `GELU`) with custom Triton epilogue kernels.

## Recognition pattern

- The convolution is already `nn.ConvTranspose3d` / ACL, not a direct custom convolution kernel.
- The editable Triton path fuses or custom-implements post ops after the convolution.
- The post chain contains standard ACL-covered ops such as `LayerNorm`, `AvgPool3d`, `GELU`.
- The custom epilogue decodes high-dimensional row ids with `//` / `%`, uses scalar-heavy per-row logic, or launches one CTA per many output rows.
- Default 3D volumes can push custom epilogue grids to or past the Ascend safe grid cap (`coreDim <= 65535`).

## Preferred optimization

Keep a tiny Triton path only when it is actually faster for small tensors; route medium/default tensors to native ACL post-processing:

```python
_TRITON_POST_MAX_NUMEL = 250_000  # tune with remote_verify

x = self.conv_transpose(x)

# Scalar add before LayerNorm is invariant: LayerNorm(x + c) == LayerNorm(x).
# If `sum_weight` is only used as a scalar before LN, omit it on the ACL path.
if x.numel() > _TRITON_POST_MAX_NUMEL:
    x = self.norm(x)
    x = self.avg_pool(x)
    return self.gelu(x)

x = fused_layernorm_tiny(x, self.norm)
y = fused_avgpool_gelu_tiny(x, self.avg_pool)
return y
```

Rationale: for medium/default 3D volumes, the custom epilogue's scalar index decode and extra Triton launches can dominate. ACL kernels for standard `LayerNorm`, `AvgPool3d`, and `GELU` are usually faster and avoid grid-cap poisoning.

## Profiling requirements

Include at least three shapes in `profile_kernels.py`:

1. `tiny_triton_path` — below the threshold; exercises the custom Triton path.
2. `medium_acl_path` — above the threshold; proves the ACL dispatch path is correct and faster.
3. `default_required` — the original `get_inputs()` shape.

If comparison providers would exceed the grid cap or poison the NPU context on the default shape, keep their columns visible but pre-skip them with a neutral `INFO ... inf` entry. Do not launch a known-toxic comparison before optimized timing.

## Cannsim interpretation

- Cannsim can compare the tiny custom sub-kernel instruction mix, but it cannot measure the benefit of avoiding full-shape launches or dispatching to ACL.
- If a realistic row-block probe is too slow or does not become safe, reduce the probe to one row (`ROWS_PER_CTA=1`, `grid=(1,1,1)`) and report it as a diagnostic micro-probe.
- Do not fabricate an optimized trace for the ACL/default path. Report that the custom Triton post-kernel was removed for that dispatch path and use `remote_verify` hardware numbers for the production decision.

## Correctness notes

- `LayerNorm(x + scalar) == LayerNorm(x)` for scalar `scalar`; removing a `sum_weight.item()` before LayerNorm is correct when the scalar is broadcast over the normalized tensor.
- Removing `.item()` also avoids a CPU/NPU synchronization in the host path.
- Keep dtype tolerance checks per dispatch path; optimized errors should remain within the usual fp32/fp16 tolerance for the operator.
