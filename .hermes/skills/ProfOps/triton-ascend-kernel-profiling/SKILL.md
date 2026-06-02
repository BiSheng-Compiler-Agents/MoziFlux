---
name: triton-ascend-kernel-profiling
description: >
  Write and run profile_kernels.py for Triton kernels on Ascend NPU hardware.
  Covers the full workflow: script structure, @perf_report usage, three-way
  comparison (torch_ref vs baseline vs optimized), correctness test, grid
  overflow guard, and when to generate the script relative to optimization work.
tags: [triton, ascend, npu, profiling, benchmarking, perf_report, kernelbench]
metadata:
  hermes:
    related_skills:
      - triton-ascend-cannsim
      - triton-ascend-optimization-patterns
      - kernel-episode-memory
---

# Triton Ascend Kernel Profiling — `profile_kernels.py`

## When to generate this script

Generate `profile_kernels.py` **after** the optimized kernel is written and
its correctness has been validated via cannsim. It belongs alongside the kernel
files in the KernelBench directory:

```
l2_<N>_<KernelName>/
├── <N>_<KernelName>.py          ← baseline Triton kernel (extracted, jit-only)
├── opt_<N>_<KernelName>.py      ← optimized kernel (all shapes)
└── profile_kernels.py           ← this script (generated last)
```

The script runs on a **real Ascend NPU** (not cannsim). cannsim is for
trace-driven optimization decisions; `profile_kernels.py` is for measuring
wall-clock latency on hardware.

---

## Script structure

### 1. Load sibling files via `importlib`

Never copy kernel code into `profile_kernels.py`. Load from the sibling files
so the profile always reflects the on-disk kernel:

```python
import importlib.util
from pathlib import Path

_DIR = Path(__file__).parent

def _load(fname):
    spec = importlib.util.spec_from_file_location(fname.stem, fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

_baseline  = _load(_DIR / "<N>_<KernelName>.py")
_optimized = _load(_DIR / "opt_<N>_<KernelName>.py")
```

### 2. Three runner functions

One per implementation:

```python
def _run_torch_ref(x, ...):
    """PyTorch built-in / ACL path — the hardware vendor reference."""
    return torch.nn.functional.relu(x) + bias   # or whatever the op is

def _run_baseline(x, ...):
    """Original Triton kernel, dispatched exactly as the baseline file intends."""
    N, C, H, W = x.shape
    y = torch.empty_like(x)
    # Guard: Ascend FFTS caps any grid dim at 65535
    # If the baseline uses a flat (N*C*H,) grid and N is large, chunk over N:
    if N * C * H <= 65535:
        _baseline_kernel[(N * C * H, w_grid)](x, y, ..., BLOCK_W=block_w)
    else:
        for n in range(N):
            _baseline_kernel[(C * H, w_grid)](x[n:n+1], y[n:n+1], ..., BLOCK_W=block_w)
    return y

def _run_optimized(x, ...):
    return _optimized.<dispatch_fn>(x, ...)
```

**Important:** Always add the 65535 grid overflow guard for the baseline.
Ascend's FFTS scheduler raises `coredim=X can't be greater than UINT16_MAX`
if any grid dimension exceeds 65535. The optimized kernel should already
handle this correctly; the baseline typically does not.

### 3. Shape table

Cover **all dispatch paths** of the optimized kernel — not just the
benchmark shape. For per-channel post-conv kernels this means:

```python
_BENCH_SHAPES = [
    # label              N    C   H_out  W_out
    ("N1-C256-14x14",    1,  256,   14,    14),  # HW=196   small → persistent path
    ("N1-C128-28x28",    1,  128,   28,    28),  # HW=784   small → persistent path
    ("N8-C64-14x14",     8,   64,   14,    14),  # N>1 small
    ("N1-C64-56x56",     1,   64,   56,    56),  # HW=3136  large → loop path
    ("N1-C96-56x56",     1,   96,   56,    56),  # non-pow2 C
    ("N1-C32-224x224",   1,   32,  224,   224),  # HW=50176 large, many tiles
    ("N128-C128-126x126",128, 128,  126,   126), # benchmark shape
]
```

Include at minimum:
- One small-HW shape (tests the persistent/startup-amortised path)
- One large-HW shape (tests the tiled loop path)
- One non-power-of-2 channel count (tests that `pid % C` is safe)
- The exact benchmark shape from the bench file

### 4. `@triton.testing.perf_report` — always use this decorator

**Never** hand-roll a benchmark loop. `perf_report` handles sweep, table,
and figure automatically:

```python
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["label"],
        x_vals=[s[0] for s in _BENCH_SHAPES],
        line_arg="mode",
        line_vals=["torch_ref", "baseline", "optimized"],
        line_names=["PyTorch / ACL", "Baseline Triton", "Optimized Triton"],
        styles=[("blue", "-"), ("red", "-"), ("green", "-")],
        ylabel="Latency (ms)",
        plot_name="<kernel_name>_perf",
        args={},
    )
)
def benchmark(label, mode):
    _, N, C, H_out, W_out = next(s for s in _BENCH_SHAPES if s[0] == label)
    x         = torch.rand(N, C, H_out, W_out, device="npu", dtype=torch.float16)
    bias      = torch.rand(C, 1, 1,            device="npu", dtype=torch.float16)
    bias_flat = bias.reshape(-1)

    if mode == "torch_ref":
        fn = lambda: _run_torch_ref(x, bias)
    elif mode == "baseline":
        fn = lambda: _run_baseline(x, bias_flat)
    else:
        fn = lambda: _run_optimized(x, bias)

    # do_bench returns seconds — perf_report labels y-axis as the ylabel above
    return triton.testing.do_bench(fn, warmup=25, rep=200, return_mode="mean")
```

**Do NOT multiply by 1e3.** `do_bench` returns seconds. `perf_report` handles
axis labelling using the `ylabel` string — just return the raw seconds value
and set `ylabel="Latency (ms)"`. The table will show correct ms values.

**Do NOT use hex colour codes or composite linestyles.** The `perf_report`
style parser only accepts:
- Plain named colours: `"blue"`, `"red"`, `"green"`, `"orange"`, etc.
- Simple linestyles: `"-"`, `"--"`, `"-."`, `":"`
- NOT accepted: `"#4C72B0"` (hex), `"-o"` (linestyle+marker combined)

Correct:
```python
styles=[("blue", "-"), ("red", "-"), ("green", "-")]
```
Wrong (will error or silently misbehave):
```python
styles=[("#4C72B0", "-o"), ("#DD8452", "-s"), ("#55A868", "-^")]
```

### 5. Unit test

Always include a correctness check that compares baseline and optimized
against the torch reference:

```python
def unit_test():
    torch.manual_seed(42)
    any_fail = False
    for label, N, C, H_out, W_out in _BENCH_SHAPES:
        x         = torch.rand(N, C, H_out, W_out, device="npu", dtype=torch.float16) * 4 - 2
        bias      = torch.rand(C, 1, 1,            device="npu", dtype=torch.float16) * 0.5
        bias_flat = bias.reshape(-1)
        ref  = _run_torch_ref(x, bias)
        base = _run_baseline(x.clone(), bias_flat)
        opt  = _run_optimized(x.clone(), bias)
        ok_b = torch.allclose(ref, base, atol=1e-2, rtol=1e-2)
        ok_o = torch.allclose(ref, opt,  atol=1e-2, rtol=1e-2)
        print(f"  {label:<24}  baseline [{'PASS' if ok_b else 'FAIL'}]  "
              f"optimized [{'PASS' if ok_o else 'FAIL'}]  "
              f"maxΔ_base={(ref-base).abs().max():.2e}  "
              f"maxΔ_opt={(ref-opt).abs().max():.2e}")
        if not ok_b or not ok_o:
            any_fail = True
    if any_fail:
        raise AssertionError("Correctness check failed")
```

Use `atol=1e-2, rtol=1e-2` for fp16 (1 ULP ≈ 1e-3 relative).

### 6. `main()` and entry point

```python
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test",  action="store_true", help="Correctness check only")
    parser.add_argument("--bench", action="store_true", help="Benchmark only")
    args = parser.parse_args()

    run_test  = args.test  or not args.bench
    run_bench = args.bench or not args.test

    if run_test:
        unit_test()
    if run_bench:
        benchmark.run(save_path=str(_DIR), print_data=True)
        # → prints table + saves <kernel_name>_perf.png automatically

if __name__ == "__main__":
    main()
```

---

## Complete file template

```python
"""
profile_kernels.py — <KernelName>

Compares three implementations on Ascend NPU hardware:
  torch_ref  : PyTorch / ACL built-in path
  baseline   : original Triton kernel (<N>_<KernelName>.py)
  optimized  : trace-optimized kernel (opt_<N>_<KernelName>.py)

Usage:
    python profile_kernels.py           # unit test + benchmark + saved figure
    python profile_kernels.py --test    # correctness only
    python profile_kernels.py --bench   # benchmark only
"""
import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import torch_npu  # noqa: F401
import triton

_DIR = Path(__file__).parent

def _load(fname):
    spec = importlib.util.spec_from_file_location(fname.stem, fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

_baseline  = _load(_DIR / "<N>_<KernelName>.py")
_optimized = _load(_DIR / "opt_<N>_<KernelName>.py")

# ── runner functions (one per implementation) ─────────────────────────────────

def _run_torch_ref(x, bias):
    return torch.nn.functional.relu(x) + bias

def _run_baseline(x, bias_flat):
    N, C, H, W = x.shape
    y       = torch.empty_like(x)
    block_w = 128 if W >= 128 else (64 if W >= 64 else 32)
    w_grid  = triton.cdiv(W, block_w)
    if N * C * H <= 65535:
        _baseline._relu_add_bias_kernel[(N * C * H, w_grid)](
            x, y, bias_flat, N, C, H, W, BLOCK_W=block_w, num_warps=4)
    else:
        for n in range(N):
            _baseline._relu_add_bias_kernel[(C * H, w_grid)](
                x[n:n+1], y[n:n+1], bias_flat, 1, C, H, W,
                BLOCK_W=block_w, num_warps=4)
    return y

def _run_optimized(x, bias):
    return _optimized._relu_add_bias_triton(x, bias)

# ── shapes ────────────────────────────────────────────────────────────────────

_BENCH_SHAPES = [
    ("N1-C256-14x14",    1, 256,  14,  14),
    ("N1-C128-28x28",    1, 128,  28,  28),
    ("N8-C64-14x14",     8,  64,  14,  14),
    ("N1-C64-56x56",     1,  64,  56,  56),
    ("N1-C96-56x56",     1,  96,  56,  56),
    ("N1-C32-224x224",   1,  32, 224, 224),
    ("N128-C128-126x126",128,128, 126, 126),
]

# ── benchmark ─────────────────────────────────────────────────────────────────

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["label"],
        x_vals=[s[0] for s in _BENCH_SHAPES],
        line_arg="mode",
        line_vals=["torch_ref", "baseline", "optimized"],
        line_names=["PyTorch / ACL", "Baseline Triton", "Optimized Triton"],
        styles=[("blue", "-"), ("red", "-"), ("green", "-")],
        ylabel="Latency (ms)",
        plot_name="<kernel_name>_perf",
        args={},
    )
)
def benchmark(label, mode):
    _, N, C, H, W = next(s for s in _BENCH_SHAPES if s[0] == label)
    x         = torch.rand(N, C, H, W, device="npu", dtype=torch.float16)
    bias      = torch.rand(C, 1, 1,   device="npu", dtype=torch.float16)
    bias_flat = bias.reshape(-1)
    fn = (_run_torch_ref  if mode == "torch_ref"  else
          _run_baseline   if mode == "baseline"   else _run_optimized)
    call = (lambda: fn(x, bias)) if mode != "baseline" else (lambda: fn(x, bias_flat))
    return triton.testing.do_bench(call, warmup=25, rep=200, return_mode="mean")

# ── unit test ─────────────────────────────────────────────────────────────────

def unit_test():
    torch.manual_seed(42)
    any_fail = False
    for label, N, C, H, W in _BENCH_SHAPES:
        x         = torch.rand(N, C, H, W, device="npu", dtype=torch.float16) * 4 - 2
        bias      = torch.rand(C, 1, 1,   device="npu", dtype=torch.float16) * 0.5
        bias_flat = bias.reshape(-1)
        ref  = _run_torch_ref(x, bias)
        base = _run_baseline(x.clone(), bias_flat)
        opt  = _run_optimized(x.clone(), bias)
        ok_b = torch.allclose(ref, base, atol=1e-2, rtol=1e-2)
        ok_o = torch.allclose(ref, opt,  atol=1e-2, rtol=1e-2)
        print(f"  {label:<24}  baseline [{'PASS' if ok_b else 'FAIL'}]  "
              f"optimized [{'PASS' if ok_o else 'FAIL'}]")
        if not ok_b or not ok_o:
            any_fail = True
    if any_fail:
        sys.exit(1)
    print("All PASS")

# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test",  action="store_true")
    parser.add_argument("--bench", action="store_true")
    args = parser.parse_args()
    run_test  = args.test  or not args.bench
    run_bench = args.bench or not args.test
    if run_test:
        unit_test()
    if run_bench:
        benchmark.run(save_path=str(_DIR), print_data=True)

if __name__ == "__main__":
    main()
```

---

## Pitfalls

- **Wrong kernel args** — always read the `@triton.jit` signature directly from
  the source file. Never assume `BLOCK_ROW` or other params exist; the original
  baseline for `_relu_add_bias_kernel` takes `N, C, H, W, BLOCK_W` only.

- **Grid overflow `coredim > UINT16_MAX`** — baseline kernels often use
  `(N*C*H,)` as gridX. For N=128, C=128, H=126 this is 2,064,384 >> 65535.
  Always guard: if `N*C*H > 65535`, loop over N and launch sub-grids of `C*H`.

- **`* 1e3` scaling** — `do_bench` returns **seconds**. Do NOT multiply to µs;
  the `perf_report` ylabel and table header should say `(ms)` and the return
  value is already in ms if you return `do_bench(...)` directly.
  The table will show values like `0.045` ms, not `45` µs.

- **Style format** — `perf_report` styles must be `(color_name, linestyle)`.
  Use plain named colours (`"blue"`, `"red"`, `"green"`) and simple linestyles
  (`"-"`, `"--"`, `"-."`). Hex colours (`"#4C72B0"`) and combined
  marker+linestyle strings (`"-o"`, `"-s"`) are not accepted.

- **Shapes must cover ALL dispatch paths — not just the benchmark shape.**
  The user will ask why only the benchmark shape was profiled. For any kernel
  with multiple dispatch paths (e.g. persistent for HW≤1024, loop for HW>1024),
  include at least: one small-HW shape, one large-HW shape, one non-power-of-2
  dimension case, and the exact benchmark shape. Profile all paths, not just one.

- **Load via importlib, never copy-paste** — if you inline the kernel code into
  `profile_kernels.py`, the profile diverges from the file on disk the moment
  you make a fix. Always load from `Path(__file__).parent / "filename.py"`.

---

## Cross-references

- Cannsim trace workflow (for deciding *which* optimizations to apply before
  writing `profile_kernels.py`): **triton-ascend-cannsim**
- Patterns discovered via trace analysis (fp16/fp32 routing, UB limits,
  hoisted zeros, persistent grids, etc.): **triton-ascend-optimization-patterns**
- Recording what worked per kernel: **kernel-episode-memory**

---

## Reference implementation

A real, working `profile_kernels.py` for the l2_1 Conv2D→ReLU→BiasAdd kernel
is stored in `references/l2_1_Conv2D_ReLU_BiasAdd_profile_kernels.py`.
Load it with:

```
skill_view("triton-ascend-kernel-profiling",
           file_path="references/l2_1_Conv2D_ReLU_BiasAdd_profile_kernels.py")
```

It demonstrates all the patterns in this skill against a real kernel:
grid overflow guard, importlib loading, perf_report decorator, shape
coverage across both dispatch paths, and unit test.
