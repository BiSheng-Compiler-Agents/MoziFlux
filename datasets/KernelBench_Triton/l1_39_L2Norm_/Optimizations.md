# Optimizations

## 1. Small-row single-load path (`N <= 8192`)

Baseline always streams each row twice: once for `sum(x*x)` and once to write `x * rsqrt(sum)`. For rows that fit in UB, the optimized path loads the row once, computes the norm, and stores from the resident vector.

```python
x = tl.load(x_ptr + row * sxm + cols * sxn, mask=mask, other=0.0).to(tl.float32)
ss = tl.sum(x * x, axis=0)
tl.store(y_ptr + row * sym + cols * syn, x * tl.rsqrt(ss), mask=mask)
```

Rationale: removes one GM read pass and cuts the small-shape hardware latency from 0.011600 ms to 0.004830 ms vs Baseline Triton1.

## 2. Contiguous large-row path

The large path keeps the original two-pass row-wise algorithm, but after `x.contiguous()` it uses direct `row * N + col` addressing instead of stride-derived column offsets.

```python
x = tl.load(x_ptr + row * N + c, mask=mask, other=0.0, care_padding=False).to(tl.float32)
acc += tl.sum(x * x, axis=0)
...
tl.store(y_ptr + row * N + c, x * inv, mask=mask)
```

Rationale: the target shape has very large rows (`D=65535`), where an attempted three-launch partial-sum decomposition was slower due extra GM traffic; direct contiguous row streaming preserved the best algorithm while trimming indexing overhead.

## 3. Grid-cap-safe row dispatch

Both kernels cap the launched programs at Ascend's FFTS limit and loop over rows inside each program.

```python
n_programs = min(M, 65535)
_kernel[(n_programs,)](..., n_programs, ...)
# kernel: while row < M: ...; row += n_programs
```

Rationale: prevents `coreDim > 65535` for valid tall inputs. `profile_kernels.py` covers this path with `overflow_rows=(70000, 16)`; baseline providers are skipped there, optimized passes.

## 4. Precision and masking

All reductions upcast to fp32 and every load/store is masked.

```python
x = tl.load(..., mask=mask, other=0.0).to(tl.float32)
acc += tl.sum(x * x, axis=0)
```

Rationale: preserves row-wise L2Norm accuracy (`UNIT_TEST PASS`, max abs error <= 1.192093e-07 across tested shapes).
