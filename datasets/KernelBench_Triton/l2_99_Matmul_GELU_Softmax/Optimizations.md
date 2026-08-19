# Optimizations

## 1. Route GEMM to ACL/Cube instead of row-wise Vector GEMM

Baseline computes every output row with vector multiply/reduce:
```python
w_vals = tl.load(w_ptrs, mask=wk_mask, other=0.0).to(tl.float32)
acc += tl.sum(w_vals * x_vals[None, :], axis=1)
```
Optimized host uses the platform GEMM implementation:
```python
z = F.linear(x, weight, bias)
```
Rationale: the target is a large 1024x8192 @ 8192x8192 GEMM. ACL dispatch uses Cube matmul kernels, while the original Triton path performs GEMM on vector lanes and becomes prohibitively slow.

## 2. Fuse small-row GELU + Softmax in one Triton row kernel

```python
z = tl.load(z_ptr + row * stride_z + offs, mask=mask, other=-float("inf")).to(tl.float32)
gel = 0.5 * z * (1.0 + tl.erf(z * 0.7071067811865476))
row_max = tl.max(tl.where(offs < N, gel, -float("inf")), axis=0)
num = tl.exp(gel - row_max)
tl.store(y_ptr + row * stride_y + offs, num / tl.sum(num, axis=0), mask=mask)
```
Rationale: for N <= 512 this avoids separate ACL activation/softmax launches and keeps the row in UB/registers.

## 3. Hardware-driven dispatch threshold

```python
if N <= 512 and z.device.type == "npu":
    _gelu_softmax_row_kernel[(B,)](...)
else:
    return torch.softmax(F.gelu(z), dim=-1)
```
Rationale: remote hardware showed the Triton epilogue wins at 512-wide rows (0.069608 ms vs 0.082914 ms ACL), but ACL wins for 2048/8192-wide rows. The threshold keeps the fast path only where it is measured faster and preserves correctness for wider rows.

## 4. Simulator-compatible cannsim harness

`cannsim_baseline/` and `cannsim_optimized/` contain sub-kernel launchers with grid=1, B=1, N=128. The baseline harness mirrors the original single-tile fast path and omits only the unsupported Ascend `cache_modifier=".cg"` hint so the simulator can compile it.
