# Optimizations

## 1. Algebraic fusion from GEMM + divide + sum to GEMV

Baseline math:
```python
y = scaling_factor * sum((x @ W.T) / 2, dim=1, keepdim=True)
```
Optimized equivalent:
```python
s_eff = weight.sum(dim=0) * (scaling_factor * 0.5)
y = torch.matmul(x, s_eff[:, None])
```
This removes the hidden-size output matrix from the hot path and reduces the operation to one GEMV over `input_size`.

## 2. Cache the effective summed weight

```python
if self._s_eff_cache is None or self._s_eff_version != self.weight._version:
    self._s_eff_cache = (self.weight.sum(dim=0) * (self.scaling_factor * 0.5)).contiguous()
```
The baseline recomputes `weight.sum(dim=0)` every forward; the optimized host interface caches it and invalidates on `Parameter._version` changes.

## 3. Hybrid production dispatch for precision and speed

```python
if K <= 4096 and self.hidden_size <= 4096:
    s_eff = self._effective_sum_weight(x)
    return torch.matmul(x, s_eff[:, None])
return ((torch.matmul(x, self.weight.t()) / 2.0).sum(dim=1, keepdim=True) * self.scaling_factor)
```
The fused GEMV fast path is used where it passes the strict 1e-3 tolerance; the target 8192x8192 case uses exact PyTorch reduction order because reassociation exceeded tolerance.

## 4. Diagnostic Triton fallback with Ascend-safe loads

```python
a = tl.load(..., mask=mask_m[:, None] & mask_k[None, :], other=0.0, care_padding=False)
b = tl.load(..., mask=mask_k[:, None] & (offs_n[None, :] == 0), other=0.0, care_padding=False)
acc = tl.dot(a, b, acc)
```
The fallback removes the unsupported `cache_modifier=".cg"`, uses masks on every load/store, applies `care_padding=False`, and uses in-place `tl.dot(a, b, acc)` for Cube execution.
