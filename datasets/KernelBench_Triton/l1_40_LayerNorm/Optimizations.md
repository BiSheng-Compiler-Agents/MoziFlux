# Optimizations

## 1. Replace atomic row reductions with private partial reductions

Baseline pass 1 launches a 2-D `(row, tile)` grid and atomically accumulates each tile into one `sums[row]` / `sumsq[row]` slot:

```python
tl.atomic_add(sums_ptr + pid_row, s)
tl.atomic_add(sumsq_ptr + pid_row, s2)
```

The optimized kernel writes each tile to a private `(row, tile)` partial buffer, eliminating atomic serialization and allowing a deterministic second-stage row reduction:

```python
out_idx = row * num_tiles + tile
tl.store(partial_sums_ptr + out_idx, s)
tl.store(partial_sumsq_ptr + out_idx, s2)
```

Rationale: target LayerNorm has `M=64*256*256`; the optimized reduction uses 16K-element tiles (256 partials/row) instead of the baseline 8K atomic tiles (512 tiles/row). Atomic updates contend on only 16 row slots; private partials preserve parallelism and reduce dispatch-side serialization while keeping FP32 reduction accuracy.

## 2. Grid-capped persistent fallback for oversized dispatch

Direct baseline grids scale as `rows * ceil(M / 8192)` and can exceed Ascend's 65535 program limit. The optimized host dispatches the direct path below the cap and persistent kernels above it:

```python
if total_tasks <= _MAX_PROGRAMS:
    _layernorm_partial_kernel[(total_tasks,)](...)
else:
    _layernorm_partial_persistent_kernel[(_MAX_PROGRAMS,)](..., total_tasks, _MAX_PROGRAMS, ...)
```

Rationale: this keeps the benchmark shape on the faster direct path while preserving correctness for valid larger row counts.

## 3. Single reciprocal for mean/variance

The optimized finalize stage computes `INV_M` once on the host and uses multiplication for both mean and variance:

```python
inv_m = float(1.0 / M)
mean = s * INV_M
var = s2 * INV_M - mean * mean
```

Rationale: Ascend scalar divide is expensive; using a host-computed reciprocal follows the known LayerNorm pattern and avoids in-kernel division.

## 4. FP32 reduction and bounded variance

All reductions upcast input to FP32, and tiny negative variance from roundoff is clamped before `rsqrt`:

```python
x = tl.load(...).to(tl.float32)
var = tl.maximum(var, 0.0)
rstd = tl.rsqrt(var + EPSILON)
```

Rationale: LayerNorm requires FP32 accumulation for fp16/bf16/fp32 correctness; clamping prevents NaN from cancellation on near-constant rows.

## 5. Larger reduction tile, unchanged apply tile

The optimized reduction uses a 16K-element tile while the normalization/apply stage keeps an 8K tile:

```python
_REDUCE_BLOCK = 16384
_APPLY_BLOCK = 8192
```

Rationale: the partial-sum stage only needs the input vector plus FP32 temporaries, so 16K fits UB and halves reduction programs/partials. The apply stage touches input, weight, bias, and output, so it remains at 8K to avoid UB pressure.
