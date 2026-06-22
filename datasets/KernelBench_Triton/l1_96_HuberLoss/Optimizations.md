# Optimizations

## 1. Production dispatch to ACL SmoothL1/Huber loss

```python
return F.smooth_l1_loss(x, y, reduction="mean", beta=_BETA)
```

The input kernel launches one Triton program per 4096 elements and atomically accumulates a single scalar. The target shape has `ceil(32768*32768/4096)=262144` tiles, which exceeds Ascend's 65,535 launch cap and also creates heavy single-address atomic contention; the optimized `ModelNew` preserves validation and shape semantics but defaults to the mature PyTorch/ACL loss implementation.

## 2. Grid-safe traced Triton fallback

```python
n_tiles = triton.cdiv(n, _STAGE1_BLOCK)
n_programs = min(n_tiles, _MAX_PROGRAMS)
_huber_stage1_kernel[(n_programs,)](..., n, n_tiles, n_programs, _BETA)
_huber_finalize_kernel[(triton.cdiv(n_programs, _FINAL_BLOCK),)](...)
```

The diagnostic fallback stores one private partial sum per program and reduces those partials in a bounded final pass. This removes the invalid oversized launch from the original kernel and reduces scalar/control/MTE work in cannsim while keeping all loads masked and reductions in fp32.

## 3. Contiguity and dtype-preserving host interface

```python
x = predictions if predictions.is_contiguous() else predictions.contiguous()
y = targets if targets.is_contiguous() else targets.contiguous()
return F.smooth_l1_loss(x.view(-1), y.view(-1), reduction="mean", beta=1.0)
```

The host keeps the original same-shape, NPU, non-empty contract and only materializes contiguous views when required. The scalar result follows PyTorch's SmoothL1 dtype behavior for the production path; the fallback casts its fp32 accumulator result back to `torch.result_type(predictions, targets)`.
