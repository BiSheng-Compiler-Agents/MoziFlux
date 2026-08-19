# Row-wise KL/Loss Reduction Dispatch Pattern

Use this for 2D row-wise scalar losses with `batchmean`/per-row reduction semantics, such as probability-input KL divergence `sum(target * (log(target) - log(pred))) / B`.

## Recognition signals

- Inputs are 2D `[B, D]`; each row is independently reduced, then divided by `B`.
- The baseline already uses one Triton program per row and a separate `row_sums.sum()` host/torch reduction.
- A mature ACL loss exists, but it may require a transformed input (for KL, `prediction.log()`) and can be slower than the custom Triton row kernel.
- Cannsim may show persistent-loop overhead at `grid=1`; hardware timing decides the dispatch threshold.

## Recommended workflow

1. **Do not assume ACL wins.** Try ACL as a candidate, but compare it against the custom Triton row kernel on hardware before replacing production dispatch.
2. Keep flattened contiguous addressing inside Triton after host-side `.contiguous()`:
   ```python
   base = row * D
   idx = base + cols
   p = tl.load(pred_ptr + idx, mask=mask, other=1.0)
   t = tl.load(targ_ptr + idx, mask=mask, other=0.0)
   ```
3. For small/medium `B*D`, test a single-launch row kernel that atomically accumulates `row_sum / B` into one scalar. This can remove the extra `row_sums.sum()` launch:
   ```python
   tl.atomic_add(out_ptr, tl.sum(acc, axis=0) / B, sem="relaxed")
   ```
4. For large many-row shapes, preserve a row-sums path if atomics regress:
   ```python
   row_sums = torch.empty((B,), device=p.device, dtype=torch.float32)
   _row_kernel[(min(B, 65535),)](...)
   return row_sums.sum() / B
   ```
5. Cap row grids and loop over rows for legality:
   ```python
   n_programs = min(B, 65535)
   for row in range(pid, B, n_programs):
       ...
   ```
6. If keeping a full-tensor fallback for diagnostics, use private partials plus a small finalize rather than unbounded launch grids.

## Reporting notes

- Include both the ACL candidate and the final custom Triton choice in the optimization rationale if ACL was rejected.
- Cannsim `grid=1` traces can look worse for persistent row-loop variants even when hardware improves; document the loop overhead and use hardware timing as the deciding metric.
- Unit tests should cover the atomic small path, row-sums large path, irregular masked tails, and any diagnostic fallback.
