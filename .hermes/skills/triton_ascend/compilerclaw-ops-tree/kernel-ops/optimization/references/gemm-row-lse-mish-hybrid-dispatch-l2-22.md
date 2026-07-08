# l2_22 GEMM + clamp + row logsumexp + Mish dispatch note

Session-specific evidence for the general GEMM ACL/Triton hybrid pattern.

## Shape and computation

- Input path: `F.linear(x, weight, bias)` then `z = clamp(y * (2 * scale_factor), clamp_min, clamp_max)`.
- Reduction/epilogue: `lse = torch.logsumexp(z, dim=1, keepdim=True)`, output `lse * (lse * tanh(softplus(lse)))`.
- Benchmark shapes: small `(B=128,I=1024,H=1024)`, medium `(256,4096,4096)`, target `(1024,8192,8192)`.

## What worked

1. Preserve ACL `F.linear` for the GEMM.
2. Clean the Triton row fallback: remove `cache_modifier='.cg'`, replace `while n < N` with `tl.range`, keep fp32 online logsumexp, and mask the final store.
3. Dispatch cleaned Triton row fallback for `H <= 4096 and B <= 65535`; dispatch ACL post-op for larger hidden sizes or row-grid overflow.

## Evidence

Cannsim post-op probe (`B=1,N=1024,BLOCK_N=1024,grid=1`):

| Metric | Baseline post-op | Optimized fallback | Change |
|---|---:|---:|---:|
| wall_cycles | 10,460 | 9,807 | -6.2% |
| PUSHQ busy_cyc | 6,361 | 6,105 | -4.0% |
| SCALAR busy_cyc | 2,339 | 1,478 | -36.8% |
| SCALARLDST busy_cyc | 3,043 | 2,447 | -19.6% |

Remote hardware:

| label | PyTorch / ACL ms | Baseline1 ms | Optimized ms | Opt vs baseline1 |
|---|---:|---:|---:|---:|
| small | 0.047480 | 0.020534 | 0.020070 | 1.023x |
| medium | 0.166260 | 0.147096 | 0.146245 | 1.006x |
| target | 1.059926 | 1.145896 | 1.054515 | 1.087x |

## Reuse guidance

For fused GEMM plus row-wise LSE/activation, source-level post-kernel cleanup can be a modest per-tile win, but production performance may hinge on dispatch threshold. Always hardware-test both the cleaned Triton row fallback and ACL `torch.logsumexp`; route large hidden reductions and grid-capped row counts to ACL unless a persistent custom path proves faster.
