# Optimizations

## 1. Contiguous row addressing

```python
x = predictions.contiguous()
y = targets.contiguous()
base = pid * D + tl.arange(0, BLOCK_SIZE)
x = tl.load(x_ptr + base, mask=mask, other=0.0).to(tl.float32)
y = tl.load(y_ptr + base, mask=mask, other=0.0).to(tl.float32)
```

Rationale: the baseline made inputs contiguous on the host but still passed and multiplied row/column strides in the kernel. The optimized kernel relies on that contiguous layout and uses a single linear row offset, reducing kernel arguments and scalar address-generation work.

## 2. Grid-cap persistent dispatch for large row counts

```python
if B <= 65535:
    _cosine_similarity_rows_contig_direct_kernel[(B,)](...)
else:
    _cosine_similarity_rows_contig_persistent_kernel[(65535,)](...)
```

Rationale: the baseline launches `grid=(B,)`, which can exceed Ascend's FFTS `coreDim <= 65535` limit for large batches. The optimized path preserves the direct launch for normal cases and adds a persistent row loop for oversized B.

## 3. Preserve fp32 reduction semantics

```python
x = tl.load(...).to(tl.float32)
y = tl.load(...).to(tl.float32)
dot = tl.sum(x * y, axis=0)
nx2 = tl.sum(x * x, axis=0)
ny2 = tl.sum(y * y, axis=0)
```

Rationale: cosine loss is reduction-sensitive; keeping fp32 accumulation matches the baseline and PyTorch reference within the unit-test tolerance.

## 4. Explicit same-device validation

```python
if predictions.device != targets.device:
    raise RuntimeError("predictions and targets must be on the same device")
```

Rationale: this catches mixed-device inputs before launching an NPU kernel while preserving the baseline 2D, equal-shape, NPU-only interface.
