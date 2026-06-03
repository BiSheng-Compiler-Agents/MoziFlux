# Remote execution lessons

## Per-variant resilience (MANDATORY for profile_kernels.py)

A provider can fail on a SPECIFIC shape (autotune "No valid triton configs",
`AttributeError`, OOM) while passing others. Naively that exception propagates
out of `do_bench` and kills the entire benchmark — you get a traceback and an
empty `results.txt` instead of a table with the other variants' numbers.

**Wrap every provider call** in both benchmark and correctness:

```python
try:
    return do_bench(lambda: _run_provider(provider, *inputs),
                    quantiles=[0.5, 0.2, 0.8])[0]
except Exception as e:
    print(f"[BENCH] {provider} failed on {label}: {type(e).__name__}: {e}")
    return float("inf")
```

This is the correct behaviour even when a baseline is genuinely broken: the
optimized kernel's column is still measured and the report is still generated.

## Correctness must check EVERY variant against torch_ref

The whole point of the multi-line profile is cross-validation; only-checking
optimized hides real bugs in the reference/baseline kernels. Isolate each
variant in try/except and print a per-variant PASS/FAIL/ERROR cell. Put the
exception MESSAGE in the cell, not just the type:

```python
cells.append(f"{name}=[ERROR:{type(e).__name__}: {str(e)[:50]}]")
```

This pattern surfaced real bugs that would have been opaque whole-run crashes:
- `base_3_*.py` uses `tl.compile_hint(...)` but compile_hint lives in `al`
  (Ascend extension), not `tl` → AttributeError on every shape.
- `base_10_*.py` fails MLIR compilation on NPU for certain shapes.

## Upload exclusion for SFTP

The remote-verify plugin's `_sftp_upload_dir` skips `remote_results/`, `build/`,
`cannsim*`, `__pycache__/`, `*.npubin` to keep upload fast. Nested
`remote_results/` from prior runs compounds across uploads. The
`fetch_remote_results.py` script calls the plugin's upload function and
inherits this automatically.

## conda run output

If a profile produces a 1-byte `results.txt` (just a newline) despite rc=0,
first check whether the script has an `if __name__ == "__main__": main()`
block. A missing block causes silent zero output (defines main() but never
calls it). This is NOT a conda buffering issue — it's a Python entry-point
bug.
