# Grid-Autotune Mismatch: Batch-Index-Dependent Error Debugging

When the ModelNew host code computes a grid using a hardcoded BLOCK size (e.g. `ceil(M/128)`)
but the Triton kernel uses `@triton.autotune` with configs that include different BLOCK_M/N
values (e.g. 64, 128, 256), the grid may not provide enough programs to compute all tiles.

## Symptom

Batch-index-dependent error magnitude: batch 0 has low error (may even pass fp16 precision),
but subsequent batches (1, 2, 3...) have progressively larger absolute errors.
This occurs because each batch gets the same number of program launches; if the grid is too
small, every batch misses the same portion of tiles, and the error magnitude varies because
each batch has different random data.

## Root Cause

The kernel's GROUP_M swizzle computes `num_pid_m = ceil(M / BLOCK_M)` where BLOCK_M is the
**autotune-chosen** value, not the hardcoded value used in the ModelNew grid computation.
When autotune picks `BLOCK_M=64` and the grid computes `ceil(M/128)`, the grid launches
half the needed programs:

```
Grid: ceil(256/128) = 2 M-tiles
Kernel: num_pid_m = ceil(256/64) = 4 M-tiles needed
→ Only 2 of 4 M-tiles computed → 50% of output = 0 (unwritten zeros)
```

## Diagnosis

1. **Per-batch error** (hardware run):
```python
for bid in range(B):
    batch_err = (c_opt[bid].float() - c_ref[bid].float()).abs().max().item()
    print(f"batch {bid} max_err={batch_err:.6f}")
```
   Error that increases with batch index is the tell.

2. **Cross-batch consistency check** — if all non-zero batches produce identical output:
```python
for bid in range(1, B):
    cross = (c_opt[0].float() - c_opt[bid].float()).abs().max().item()
    print(f"c_opt[0] vs c_opt[{bid}] diff={cross:.6f}")
```
   If c_opt[0] ≠ c_opt[1] but both are wrong, the error is from tile undercoverage
   (different data per batch → different error per unwritten tile location).

## Fix

Use a conservative grid that covers the smallest BLOCK in the autotune configs:

```python
# Smallest BLOCK_M/N across all autotune configs
MIN_BLOCK_M = min(c["BLOCK_M"] for c in autotune_configs)
MIN_BLOCK_N = min(c["BLOCK_N"] for c in autotune_configs)
grid_m = triton.cdiv(M, MIN_BLOCK_M)
grid_n = triton.cdiv(N, MIN_BLOCK_N)
grid = (grid_m * grid_n, 1)
```

Extra programs are harmless: their tile offsets are beyond the matrix bounds,
all masks evaluate to false, and no stores are performed.

## For Batched Broadcasting (B > 1)

When one operand has B=1 and the other has B>1, avoid the 3D batched kernel entirely.
Use a per-batch sequential 2D matmul with conservative grid:

```python
def _matmul_2d(A, B, M, N, K):
    C = torch.zeros(M, N, device=A.device, dtype=A.dtype)
    grid = (triton.cdiv(M, 64) * triton.cdiv(N, 64), 1)
    _kernel[grid](A, B, C, 1, M, N, K, ...)
    return C

# In forward:
results = [_matmul_2d(a[bid], b_single, M, N, K) for bid in range(B)]
```

This trades batch parallelism for correctness guarantee. For broadcasting cases
(B=1 × B>1 or B>1 × B=1), the overhead is negligible since one operand is tiny.

## Confirmed Episodes

- l1_10 3D tensor matrix multiplication (June 2026, episode 72):
  Original hardcoded `ceil(M/128)` in ModelNew forward with 9-config autotune
  (configs had BLOCK_M=64, 128, 256). Batch 0 passed fp16 precision by coincidence;
  batches 1-3 had 49-60 element error. Fixed by per-batch 2D loop with `ceil(M/64)` grid.
