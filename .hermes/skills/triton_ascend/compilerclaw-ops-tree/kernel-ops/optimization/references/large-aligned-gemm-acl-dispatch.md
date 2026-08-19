# Large aligned GEMM: ACL production dispatch + Triton fallback

Use this when optimizing `nn.Linear`/GEMM-dominated operators on Ascend where the target production shape is a large, aligned FP32/BF16/FP16 GEMM and the custom Triton value is mainly in non-standard epilogues or irregular fallback coverage.

## Pattern

1. **Keep the production aligned path on ACL** when hardware profiling shows `torch.nn.functional.linear` / ACL is already faster than the editable Triton GEMM.
2. **Keep a legal Triton fallback** for non-aligned or shape-regime coverage, especially if the deliverable requires an optimized Triton kernel and cannsim trace data.
3. **Profile both paths explicitly**: include at least one aligned production shape and one irregular fallback shape in `profile_kernels.py` unit tests and benchmark labels.
4. **Keep cannsim claims scoped**: sub-kernel cannsim can compare the Triton fallback against the editable Triton baseline, but hardware latency for the production path comes from `remote_verify` / real NPU profiling.

## Host dispatch skeleton

```python
if (M % BLOCK_M) == 0:
    return torch.nn.functional.linear(x_fp32, self.matmul.weight, self.matmul.bias) * scale

# Irregular / fallback path: custom Triton kernel.
w_kn = self._weight_kn()  # cached W.transpose(0, 1).contiguous()
_linear_kernel[grid](x_fp32, w_kn, bias, out, ...)
```

## Triton fallback improvements

- Cache a contiguous `[K, N]` weight layout to avoid per-call transpose and strided B loads.
- Use `tl.dot(a, b, acc)` in-place accumulation rather than `acc += tl.dot(...)`.
- Prefer a 1-D AI-core-capped grid with an intra-core tile loop over a large 2-D tile grid.
- Use `tl.range` for the K loop and `al.compile_hint(acc, "dot_pad_only_k")` when M/N are 16-aligned.

## Reporting requirements

- In `performance_report.md`, separate:
  - cannsim fallback trace deltas (baseline Triton vs optimized Triton fallback), and
  - hardware production latencies (ACL path vs Baseline Triton1/2 vs PyTorch / ACL).
- In `review.md`, mark the fallback path as tested if the profiler includes a non-aligned shape that triggers it.

## Pitfalls

- Do not claim cannsim sub-kernel wall cycles represent the full aligned ACL production dispatch; simulation only covers the compiled Triton sub-kernel.
- Do not leave the fallback untested just because the required benchmark shape uses ACL.
- If the irregular Triton fallback is slower than ACL but correct, document it as coverage rather than as the production fast path.
