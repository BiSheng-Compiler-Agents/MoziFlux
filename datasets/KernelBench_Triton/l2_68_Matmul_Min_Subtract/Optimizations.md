# Optimizations Applied

## 1. Cache the transposed weight and NPU-side parameters

**Baseline pattern**
```python
weight = self.linear.weight.detach().to(device=x.device, dtype=x.dtype)
w_c = weight.transpose(0, 1).contiguous()
```
This re-materializes a 16384 x 16384 transposed contiguous weight every forward pass at the target shape.

**Optimized pattern**
```python
if key != self._cached_key:
    weight = self.linear.weight.detach().to(device=x.device, dtype=x.dtype)
    self._cached_weight_t = weight.transpose(0, 1).contiguous()
    self._cached_bias = self.linear.bias.detach().to(device=x.device, dtype=x.dtype).contiguous()
    self._cached_constant = self.constant.detach().to(device=x.device, dtype=x.dtype).contiguous()
```
Rationale: the model is inference-only (`requires_grad` inputs are rejected), so caching avoids repeated CPU/NPU conversion and a large transposition copy while preserving correctness for stable model parameters.

## 2. Increase K tile from 32 to 64

**Baseline**
```python
"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32
```

**Optimized**
```python
BLOCK_M = 64
BLOCK_N = 128
BLOCK_K = 64
```
Rationale: 64x128x64 remains within UB/L1 headroom and halves the K-loop trip count for the target `K=16384`, reducing loop/control and repeated MTE/FLOWCTRL work while preserving 16x16 Cube granularity.

## 3. In-place Cube accumulation

**Baseline**
```python
acc += tl.dot(x, w, out_dtype=tl.float32)
```

**Optimized**
```python
acc = tl.dot(x, w, acc, out_dtype=tl.float32)
```
Rationale: `tl.dot(a, b, acc)` accumulates inside Cube hardware and avoids an extra fp32 temporary tile and vector add/store path.

## 4. Index-recomputed `tl.range` loop and safe padding hint

**Optimized**
```python
for k_start in tl.range(0, K, BLOCK_K):
    k_idxs = k_start + offs_k
    x_ptrs = X_ptr + offs_m[:, None] * stride_xm + k_idxs[None, :] * stride_xk
    w_ptrs = WT_ptr + k_idxs[:, None] * stride_wk + offs_n[None, :] * stride_wn
    x = tl.load(x_ptrs, mask=mask_m[:, None] & k_mask[None, :], other=0.0, care_padding=False)
    w = tl.load(w_ptrs, mask=k_mask[:, None] & mask_n[None, :], other=0.0, care_padding=False)
```
Rationale: recomputing offsets from `k_start` avoids mutable pointer recurrence inside the loop, and `care_padding=False` is safe for masked matmul lanes because padded values are zero and only contribute zero to `tl.dot`.

## Verification summary

- `remote_verify` correctness: `UNIT_TEST PASS`.
- Optimized target max absolute difference: `3.57628e-06`.
- cannsim sub-kernel: baseline `11628` cycles vs optimized `9365` cycles for the same `M=64,N=128,K=64` micro-probe.
- Hardware target latency: Baseline Triton1 `71.961739 ms`; Optimized Triton `39.232811 ms`.
