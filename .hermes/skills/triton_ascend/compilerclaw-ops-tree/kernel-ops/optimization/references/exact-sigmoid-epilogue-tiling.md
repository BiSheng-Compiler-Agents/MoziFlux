# Exact sigmoid post-linear epilogues: tiling and rewrite checks

## Trigger

Use this note for GEMM/Linear outputs followed by an exact sigmoid/SILU-style elementwise residual/scale epilogue, e.g.:

```python
z = linear(x)
y = z + scale * sigmoid(z)
# or y = z * sigmoid(z) * scale
```

## Lessons

- Exact sigmoid epilogues are usually dominated by vector `exp`/`div` (`RV_VEXP`, `RV_VDIV`, RVECEX) in cannsim.
- Do **not** assume a smaller `BLOCK_SIZE` is faster because raw sub-kernel wall cycles dropped. Normalize by elements/cycle or cycles/element when block sizes differ.
- Large contiguous tiles such as `BLOCK_SIZE=16384` can have much better normalized throughput than `4096` even if the 4096 trace has lower absolute cycles.
- An algebraic `exp2(-x * log2(e))` sigmoid rewrite can regress on Ascend because it adds vector multiply work and may not reduce `div/exp` pressure enough. A/B it before adopting.
- Keep direct dispatch for `cdiv(n_elements, BLOCK_SIZE) <= 65535`; add a separate persistent fallback only beyond the grid cap.
- Force-test the persistent path by temporarily lowering the grid cap in `profile_kernels.py` instead of allocating huge tensors.

## Verification pattern

Compare cannsim traces using the same mathematical work when possible. If tile sizes differ, report normalized metrics:

| Metric | Why |
|---|---|
| cycles/element | Avoids accepting smaller tiles that only reduce raw work per program |
| RVECEX per element | Shows exact sigmoid vector pressure |
| `RV_VEXP`, `RV_VDIV` totals | Confirms whether a rewrite actually removed the bottleneck |

Production benchmark still needs `remote_verify`; cannsim sub-kernel traces do not show full FFTS dispatch effects.
