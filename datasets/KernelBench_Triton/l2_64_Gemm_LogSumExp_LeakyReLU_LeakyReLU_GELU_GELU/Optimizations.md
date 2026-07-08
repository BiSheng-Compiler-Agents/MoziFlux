# Optimizations Applied

## 1. Hybrid production dispatch to CANN/ACL for the large GEMM + row LogSumExp regime

```python
x = self.linear(x)
if _USE_ACL_LSE and x.shape[1] >= _ACL_LSE_MIN_N:
    return _post_lse_acl(x, self.neg_slope)
return _post_lse_triton(x, self.neg_slope)
```

Rationale: the default problem is a large `F.linear` followed by standard row-wise `logsumexp`, two LeakyReLU operations, and two exact GELU operations. Hardware profiling showed CANN/ACL is only preferable at the default large `N=8192` target; small and medium shapes keep the Triton fallback. The Triton fallback trace is dominated by `PUSHQ`/front-end activity rather than arithmetic throughput.

## 2. Kept an explicit Triton fallback with direct and persistent row dispatch

```python
if force_persistent or B > _MAX_PROGRAMS:
    _rowwise_lse_leaky_gelu2_persistent[(min(B, _MAX_PROGRAMS),)](...)
else:
    _rowwise_lse_leaky_gelu2_direct[(B,)](...)
```

Rationale: the original kernel launched one program per row and had no FFTS grid-cap handling. The optimized fallback preserves the direct path for normal `B <= 65535` shapes and adds a persistent row loop for larger batches, avoiding illegal grid sizes without routing normal shapes through the slower persistent loop.

## 3. Converted the fallback loop to `tl.range`

```python
for n0 in tl.range(0, N, BLOCK_N):
    offs = n0 + r
    vals = tl.load(row_ptr + offs * stride_xn, mask=offs < N, other=-float("inf")).to(tl.float32)
```

Rationale: `tl.range` is the Ascend-preferred structured loop form and avoids Python-loop lowering hazards in more complex kernels. Cannsim shows this row kernel remains front-end/PUSHQ dominated, so the main production win is host dispatch selection rather than a sub-kernel instruction-count change.

## 4. Preserved numerics exactly

```python
y = torch.logsumexp(x, dim=1, keepdim=True)
y = F.leaky_relu(y, negative_slope=neg_slope)
y = F.leaky_relu(y, negative_slope=neg_slope)
y = F.gelu(y)
y = F.gelu(y)
```

Rationale: `torch.logsumexp` keeps max-subtraction stability, and `F.gelu` defaults to the exact erf-based formulation matching the Triton source. No tanh GELU approximation or reassociation of the GEMM reduction was introduced.
