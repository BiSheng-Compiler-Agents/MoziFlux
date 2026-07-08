# Manual timing and large-output validation notes

Use this reference when `profile_kernels.py` has to fall back from `triton.testing.do_bench` or when validating very large elementwise tensors.

## Manual timing fallback

If `do_bench` is unavailable or produces implausibly tiny timings on `torch_npu`, use explicit per-iteration synchronization. Synchronizing only after the whole repetition loop can under-time async Triton launches.

```python
def _sync():
    try:
        torch.npu.synchronize()
    except Exception:
        pass


def _bench_callable(fn):
    for _ in range(3):
        out = fn()
        _sync()
        # Optional for kernels suspected of lazy or deferred writes; avoid in hot path unless needed.
        # _ = float(out.reshape(-1)[0].item())
        del out
    t0 = time.perf_counter()
    for _ in range(10):
        out = fn()
        _sync()
        del out
    return (time.perf_counter() - t0) * 1000.0 / 10.0
```

Keep `@triton.testing.perf_report` as the table/report wrapper even when the timing body uses manual timing.

## Correctness for huge tensors

For tensors large enough to trigger grid-capping or offset-chunking logic, test both:

1. the exact original oversized shape, and
2. at least one dispatch-safe direct shape.

If a custom huge-tensor Triton path uses runtime pointer offsets, chunk offsets, or persistent grid-stride loops, add targeted samples from the start, middle, and end of the output before trusting the path. If those samples disagree with the reference, prefer a correctness-preserving fallback over reporting speed for an invalid kernel.
