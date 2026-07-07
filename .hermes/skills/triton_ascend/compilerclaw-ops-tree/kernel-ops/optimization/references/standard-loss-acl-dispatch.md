# Standard Loss Reduction ACL Dispatch Pattern

Use this for scalar loss reductions covered by PyTorch/ACL, such as MSELoss-like `mean((x-y)^2)` or SmoothL1/Huber `mean(huber(x-y))` reductions, when the custom Triton baseline is a whole-tensor reduction with many programs contending on one scalar accumulator.

## Recognition signals

- The operator is a standard PyTorch loss/reduction with a mature ACL implementation (`F.mse_loss`, `F.smooth_l1_loss`, etc.).
- The custom Triton kernel uses one program per tile plus `tl.atomic_add` into a single scalar output.
- Target shapes can exceed Ascend's launch cap, e.g. `ceil(n / BLOCK_SIZE) > 65535`.
- Cannsim sub-kernel shows scalar/PUSHQ/MTE stalls from reduction bookkeeping rather than useful fused work.

## Recommended production pattern

Preserve the original `ModelNew` interface and validation, then default production dispatch to ACL:

```python
import torch.nn.functional as F

class ModelNew(nn.Module):
    def forward(self, predictions, targets):
        # keep baseline shape/device/dtype/non-empty validation
        x = predictions if predictions.is_contiguous() else predictions.contiguous()
        y = targets if targets.is_contiguous() else targets.contiguous()
        x = x.view(-1)
        y = y.view(-1)
        return F.smooth_l1_loss(x, y, reduction="mean", beta=1.0)
        # or: return F.mse_loss(x, y, reduction="mean")
```

Only keep a Triton fallback if it is useful for cannsim tracing or for a non-default diagnostic path:

```python
if not getattr(self, "_use_triton_fallback", False):
    return F.smooth_l1_loss(x, y, reduction="mean", beta=1.0)
# optional traced fallback below
```

## Triton fallback shape, if needed

- Stage 1: cap launch programs with `n_programs = min(cdiv(n, BLOCK), 65535)` and loop over tiles inside each program.
- Store one private partial per program instead of atomically updating the scalar output.
- Finalize: reduce partials in one or a few programs; keep final atomic count tiny.

```python
n_tiles = triton.cdiv(n, BLOCK)
n_programs = min(n_tiles, 65535)
_stage1[(n_programs,)](..., n_tiles, n_programs, BLOCK_SIZE=BLOCK)
_finalize[(triton.cdiv(n_programs, FINAL_BLOCK),)](...)
```

## Profiling/reporting notes

- Include `PyTorch / ACL`, baseline Triton, read-only `base_*.py`, and optimized providers.
- For source baselines whose grid exceeds 65,535, pre-skip with a neutral `grid_guard` line and `inf` benchmark cell so the NPU context is not poisoned.
- Use cannsim on the baseline and fallback micro-probes to document trace bottlenecks; hardware timing decides whether ACL or fallback is the production path.
- Do not record this as a per-kernel story in SKILL.md; put exact trace/timing values in `performance_report.md` and/or an episode.
