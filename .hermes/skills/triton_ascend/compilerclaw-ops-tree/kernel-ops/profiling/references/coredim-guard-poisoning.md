# Ascend coreDim guard poisoning in profilers

When a comparison provider can launch a grid above Ascend `coreDim <= 65535`, guard it before **both** correctness tests and benchmarks. A failed launch can poison the current NPU process, making later providers report bogus mismatches or runtime errors even if their kernels are correct.

## Pattern

Implement the guard in the shared provider dispatch wrapper, not only inside the benchmark path:

```python
def _run_provider(key, x):
    if key == "baseline1" and _would_overflow_baseline1_grid(x):
        raise RuntimeError("SKIP coreDim_guard: baseline1 grid exceeds 65535")
    if key == "baseline2" and _would_overflow_baseline2_grid(x):
        raise RuntimeError("SKIP coreDim_guard: baseline2 grid exceeds 65535")
    return _model(key, x.device.type)(x)
```

In unit tests, treat read-only baseline guard exceptions as `INFO`/`SKIP`, but still require optimized correctness. In benchmarks, keep the provider column and return `inf`.

## Derive guard formulas from each provider's actual grid

Do not use one generic element-count guard for all providers. Read the provider host code and mirror its launch grid:

- Row-wise NCHW RMSNorm baseline: `B * H * W > 65535`
- HW-block RMSNorm baseline with `BLOCK_HW=128`: `B * cdiv(H*W, 128) > 65535`
- Elementwise direct path: `cdiv(n_elements, BLOCK_SIZE) > 65535`

Guarding the exact grid prevents an invalid comparison kernel from poisoning the process before the optimized provider runs.

## Explicit numeric correctness after guarded failures

After a launch failure has been avoided, compute explicit `max_abs` and `max_rel` against the reference and gate optimized correctness on those scalars. This makes the printed failure mode diagnosable and avoids hiding corrupted-context artifacts behind a single `torch.allclose` boolean.
