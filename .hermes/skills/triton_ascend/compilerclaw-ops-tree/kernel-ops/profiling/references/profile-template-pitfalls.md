# profile_kernels.py Pitfalls

## Correctness gates benchmark meaning

Benchmark numbers are only meaningful for providers that match `PyTorch / ACL` on the same `_BENCH_SHAPES`. If `Baseline Triton1` (`<N>_*.py`) is editable and produces NaNs or large errors, debug and fix it before comparing timings. If a read-only comparison provider fails, keep its column but return `inf` for benchmark cells and label the unit-test line as `INFO`/`SKIP`, not `FAIL`, so optimized verification can still pass.

## Match source input contract

Build `_make_inputs()` from the original `<N>_*.py` `get_inputs()` contract: shape, dtype, layout, and distribution. Do not switch `torch.rand` to `torch.randn` or fp16 to fp32 unless the source contract does. Several matmul baselines only accept fp16/bf16 despite `torch.rand` defaulting to fp32 on CPU examples; profile with the NPU-supported dtype.

## Ascend `tl.dot` operand dtype

For fp16/bf16 matmul kernels on Ascend, load operands in native dtype and use an fp32 accumulator:

```python
acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
a = tl.load(a_ptrs, mask=a_mask, other=0.0)
b = tl.load(b_ptrs, mask=b_mask, other=0.0)
acc += tl.dot(a, b)          # or: acc = tl.dot(a, b, acc)
```

Do not upcast operands before `tl.dot`:

```python
# Bad on Ascend for fp16/bf16 matmul: can produce large errors or NaNs
a = tl.load(...).to(tl.float32)
b = tl.load(...).to(tl.float32)
acc += tl.dot(a, b, allow_tf32=False)
```

This bug appeared in multiple batched/4D matmul baselines; the fix was removing `.to(tl.float32)` on dot operands while keeping the accumulator fp32.

## CoreDim poisoning

Before launching a Triton provider, estimate the most conservative grid from the smallest autotune block sizes. If it exceeds Ascend `coreDim <= 65535`, do not launch it; return `inf` for that provider/shape. A failed launch poisons the current NPU context, making later providers fail with `507000` / `event recorder null`.

## `BLOCK_SIZE` conflict with `@triton.autotune` on Ascend

When calling an `@triton.autotuned` kernel from Python, do NOT pass `BLOCK_SIZE` as a keyword argument:

```python
# WRONG — raises ValueError: Conflicting meta-parameters: BLOCK_SIZE
_swish_kernel_autotuned[grid](x, y, n, n_pow2, BLOCK_SIZE=256)

# CORRECT — autotune sets BLOCK_SIZE internally
_swish_kernel_autotuned[grid](x, y, n, n_pow2)
```

`@triton.autotune` clears non-config kwargs before setting config values. Since `BLOCK_SIZE` is an autotune config key, passing it explicitly as a kwarg creates a conflict. The grid must be conservative (based on smallest block in configs); autotune selects the actual `BLOCK_SIZE` at compile time.

## Persistent dispatch requires a SEPARATE `@triton.jit` function

When implementing two-path dispatch (direct for small inputs, persistent for large inputs), the persistent kernel must be a **separate** `@triton.jit` function — not the same one decorated with `@triton.autotune`. Reasons:

1. `@triton.autotune` compiles multiple kernel variants with different `BLOCK_SIZE` values. Persistent kernels use a fixed `BLOCK_SIZE`.
2. The persistent kernel receives `n_programs` as a runtime arg (not `tl.constexpr`).
3. The persistent work loop is: `for tile_id in range(pid, n_tiles, n_programs)` — `n_programs` is only known at Python launch time, not at JIT compile time.

```python
# Separate persistent kernel — NOT decorated with @triton.autotune
@triton.jit
def _kernel_persistent(x_ptr, y_ptr, n_elements, n_programs, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    n_tiles = tl.cdiv(n_elements, BLOCK_SIZE)
    for tile_id in range(pid, n_tiles, n_programs):
        offs = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements
        x = tl.load(x_ptr + offs, mask=mask, other=0.0)
        tl.store(y_ptr + offs, compute(x), mask=mask)
```

The `direct` path uses the `@triton.autotuned` function; the persistent path uses the above. Selection happens in Python based on `cdiv(n_elements, min_block) > 65535`.

## No subprocess wrappers by default

Use the canonical single-process template with importlib-loaded providers, try/except cells, `x_names=["label"]`, and `perf_report`. Do not isolate each provider/shape in subprocesses unless explicitly asked; it hides the standard profiler structure and breaks parser expectations.

## Compile hints

Do not encode permanent negative rules about `tl.compile_hint` or `al.compile_hint` from old environment-specific failures. Verify behavior on the active Triton-Ascend stack. If a provider MLIR-aborts for every shape, handle that provider as `inf` without launching it; do not generalize the failure to all kernels.
