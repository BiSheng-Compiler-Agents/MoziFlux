# Optimizations Applied

## 1. Contiguous-only kernel signature

Baseline carries general strides through the device kernel even though the host already calls `x.contiguous()` and allocates contiguous `y`.

```python
# baseline
x_row_ptr = x_ptr + pid * stride_xm
tl.load(x_row_ptr + offs * stride_xn, ...)

# optimized
row_base = row * N
tl.load(x_ptr + row_base + offs, mask=mask, other=0.0, care_padding=False)
```

Rationale: removes stride arguments and per-access stride multiplies from the hot loop while preserving the baseline host behavior of returning a contiguous output.

## 2. Tensor accumulator for row L1 sum

```python
acc = tl.zeros((1,), dtype=tl.float32)
acc += tl.sum(tl.abs(x.to(tl.float32)), axis=0, keep_dims=True)
inv = 1.0 / acc
```

Rationale: the baseline scalar accumulator (`tl.zeros((), ...)`) causes scalar load/store spill traffic across loop-carried reductions on Ascend. A one-lane tensor accumulator keeps the reduction state in vector registers and cuts `SCALARLDST` operations in cannsim.

## 3. Simulator/compiler-safe memory operations

```python
tl.load(ptr + offsets, mask=mask, other=0.0, care_padding=False)
tl.store(out + offsets, value, mask=mask)
```

Rationale: all accesses remain masked for Ascend OOB safety, `care_padding=False` avoids redundant padding checks, and the optimized kernel avoids the baseline `.cg` cache modifier that is unsafe for Triton-Ascend compilation.

## 4. FFTS grid cap for row dispatch

```python
n_programs = min(B, 65535)
_l1norm_row_kernel_opt[(n_programs,)](..., n_programs, BLOCK_SIZE=block_size)

# device
row = tl.program_id(0)
while row < B:
    ...
    row += n_programs
```

Rationale: the benchmark shape has `B=32768`, but the host interface should not crash if a caller provides more than 65535 rows. The persistent row loop preserves correctness without changing the normal fast path.

## 5. Ascend launch metadata cleanup

```python
_l1norm_row_kernel_opt[(n_programs,)](..., num_warps=4, num_stages=2)
```

Rationale: `num_stages=2` avoids unsupported/high-stage Ascend behavior while keeping the original 4x unrolled row loop for the target large-`N` regime.
