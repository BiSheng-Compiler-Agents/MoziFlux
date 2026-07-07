# Comparison-provider pre-skip pattern

When a comparison provider is known to be invalid for a shape (compile-time VF stack overflow, unsupported API, or `coreDim > 65535`), pre-skip it before calling the provider. Do this in both `unit_test()` and `benchmark()`.

Why: executing a known-broken comparison provider can emit large MLIR `ERROR` transcripts or poison the NPU context before the optimized provider runs. A later `UNIT_TEST PASS` for optimized may still be reported as `test_passed=false` by tooling if the captured output contains those failures.

Pattern:
```python
def _provider_unavailable(key, shape):
    if key == "baseline1" and KNOWN_COMPILE_FAIL:
        return "known_vf_stack_compile_failure"
    if key == "baseline2" and grid_overflows(shape):
        return "coreDim_guard"
    return None

# unit_test
reason = _provider_unavailable(key, tuple(x.shape))
if reason:
    print(f"TEST {key} {label} SKIP {reason}")
    continue

# benchmark
reason = _provider_unavailable(provider, shape)
if reason:
    print(f"INFO {provider} {label} INF {reason}")
    return float("inf")
```

Keep the comparison provider column in the `@perf_report` table, but do not execute a provider that is already known to be unshippable for that shape. Optimized correctness remains the gate.