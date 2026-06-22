# Optimizations for 90_cumprod

## 1. Replaced scalar per-element row scan with block prefix scan

Baseline walks every element with a dynamic `while i < N` loop:

```python
carry = 1.0
i = 0
while i < N:
    v = tl.load(x_row_ptr + i * stride_xn)
    carry = carry * v
    tl.store(y_row_ptr + i * stride_yn, carry)
    i += 1
```

Optimized kernel processes each row in `BLOCK=512` chunks using `tl.cumprod` in vector lanes, with a scalar carry only between chunks:

```python
vals = tl.load(x_row_ptr + offs * stride_xn, mask=mask, other=1.0).to(tl.float32)
scan = tl.cumprod(vals, axis=0)
out = scan * carry
tl.store(y_row_ptr + offs * stride_yn, out, mask=mask)
block_last = tl.sum(tl.where(rel == last_rel, scan, 0.0), axis=0)
carry *= block_last
```

Rationale: this preserves exact left-to-right cumprod semantics while reducing loop iterations from `N` scalar iterations to `ceil(N / 512)` vector prefix-scan chunks.

## 2. Added complete masks and neutral padding

```python
mask = offs < N
vals = tl.load(..., mask=mask, other=1.0).to(tl.float32)
tl.store(..., out, mask=mask)
```

Rationale: Ascend requires masked boundary memory operations; `other=1.0` is the multiplicative identity, so tail lanes do not affect the selected final prefix product.

## 3. Added persistent row-grid fallback

```python
if rows <= _MAX_PROGRAMS:
    _cumprod_rowwise_block_kernel[(rows,)](...)
else:
    n_programs = _MAX_PROGRAMS
    _cumprod_rowwise_block_persistent_kernel[(n_programs,)](..., n_programs, BLOCK=_BLOCK)
```

Rationale: the source target has 32,768 rows, but the host now also supports valid inputs with more than Ascend's 65,535 1D launch programs by looping over rows inside a capped grid.

## 4. Preserved generic `dim` support

The host keeps the original `permute(...).contiguous() -> flatten rows -> inverse permute` structure, so any valid dimension accepted by the baseline remains supported.
