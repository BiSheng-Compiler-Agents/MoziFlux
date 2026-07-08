# Post-linear Swish epilogue: direct sigmoid rewrite

Use this when a KernelBench-style operator is `F.linear`/GEMM followed by Swish/SILU and a scalar multiply, and the custom Triton work is only the contiguous epilogue.

## Pattern

Baseline ports often use a branch-stable sigmoid form:

```python
z = tl.exp(-tl.abs(x))
s = tl.where(x >= 0, 1.0 / (1.0 + z), z / (1.0 + z))
out = x * s * scale
```

For typical fp16/bf16 Swish epilogues, test the simpler fp32 direct sigmoid:

```python
x = tl.load(ptr + offs, mask=mask, other=0.0, care_padding=False).to(tl.float32)
sigmoid = 1.0 / (1.0 + tl.exp(-x))
out = x * sigmoid * scale
tl.store(out_ptr + offs, out, mask=mask)
```

## Why it can help

The branch-stable form may issue two vector-divide groups plus `abs`/`where` work. On one same-tile cannsim comparison (`BLOCK_SIZE=8192`, one program), direct sigmoid reduced wall cycles `4627 -> 4041`, `RVECEX 1141 -> 562`, and `PUSHQ 1186 -> 608` while preserving correctness against `F.silu` within fp16 tolerance.

## Guardrails

- Upcast to fp32 before `exp/div/mul`, then store through the output pointer dtype.
- This is an approximation/semantic choice relative to the stable branch form; verify against the PyTorch reference on negative and positive values, not just random `[0,1)` inputs.
- Keep shape-aware block dispatch. A fixed large block can regress small/medium outputs even if it wins at the default shape.
- Use direct dispatch below the 65535-program cap and persistent fallback only above it; force-test the persistent path by temporarily lowering the cap in `profile_kernels.py`.
- For cannsim, compare equal element counts/block sizes, or report normalized cycles per element.
