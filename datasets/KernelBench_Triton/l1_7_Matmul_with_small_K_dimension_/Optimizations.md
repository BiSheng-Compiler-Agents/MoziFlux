# Optimizations Applied

## 1. Removed unsafe/oversized autotune configs for the benchmark shape

```python
configs=[
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 32}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=2),
]
```

Rationale: the baseline autotuner can select configs whose full-grid launch reaches `coreDim=65536`, which exceeds Ascend's `<=65535` launch limit on the benchmark shape. The optimized config set keeps the benchmark grid at 32768 programs and avoids the runtime launch failure.

## 2. Replaced Python `range` K-loop with `tl.range`

```python
num_k_iters = tl.cdiv(K, BLOCK_K)
for k_idx in tl.range(0, num_k_iters):
    rk = rk_base + k_idx * BLOCK_K
```

Rationale: `tl.range` exposes the loop to Triton-Ascend scheduling and avoids Python-style loop lowering in the device body.

## 3. In-place Cube accumulation

```python
acc = tl.dot(a, b, acc)
```

Rationale: accumulating through the `tl.dot(..., acc)` operand avoids materializing an extra fp32 tile from `acc += tl.dot(...)` and is the preferred Ascend matmul pattern.

## 4. Hoisted masks and `care_padding=False`

```python
row_mask = rm[:, None] < M
col_mask = rn[None, :] < N
a = tl.load(a_ptrs, mask=row_mask & k_mask[None, :], other=0.0, care_padding=False)
b = tl.load(b_ptrs, mask=k_mask[:, None] & col_mask, other=0.0, care_padding=False)
```

Rationale: row/column masks are invariant across K tiles. `care_padding=False` is safe for matmul because out-of-bounds inputs are zero and contribute zero to the dot product.

## 5. Ascend Cube padding hint

```python
al.compile_hint(a, "dot_pad_only_k")
al.compile_hint(b, "dot_pad_only_k")
```

Rationale: only the K dimension needs Cube padding for these matmul tiles; this hint prevents unnecessary M/N padding work where supported by Triton-Ascend.
