# Kernel Hardware Profiling [LEAF NODE]

Write and run `profile_kernels.py` for Triton kernels on Ascend NPU hardware.
Covers the full workflow: script structure, `@perf_report` usage, three-way comparison
(torch_ref vs baseline vs optimized), correctness test, grid overflow guard.
Runs on a **real Ascend NPU** (not cannsim) — for final wall-clock latency measurement.

## When to generate this script

Generate `profile_kernels.py` **after** the optimized kernel is written and its correctness
has been validated via cannsim. It belongs alongside the kernel files:

```
l2_<N>_<KernelName>/
├── <N>_<KernelName>.py          ← baseline Triton kernel (extracted, jit-only)
├── opt_<N>_<KernelName>.py      ← optimized kernel (all shapes)
└── profile_kernels.py           ← this script (generated last)
```

---

## Workflow

### Step 1: Load sibling files via `importlib`

Never copy kernel code into `profile_kernels.py`. Load from the sibling files:

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

### Step 2: Three runner functions

One per implementation:

```python
def _run_torch_ref(x, ...):
    """PyTorch built-in / ACL path — the hardware vendor reference."""
    return torch.nn.functional.relu(x) + bias

def _run_baseline(x, ...):
    """Original Triton kernel, dispatched exactly as the baseline file intends."""
    N, C, H, W = x.shape
    y = torch.empty_like(x)
    # Guard: Ascend FFTS caps any grid dim at 65535
    if N * C * H <= 65535:
        _baseline._kernel[(N * C * H, w_grid)](x, y, ..., BLOCK_W=block_w)
    else:
        for n in range(N):
            _baseline._kernel[(C * H, w_grid)](x[n:n+1], y[n:n+1], ..., BLOCK_W=block_w)
    return y

def _run_optimized(x, ...):
    return _optimized._dispatch_fn(x, ...)
```

**Always add the 65535 grid overflow guard for the baseline.**
Ascend's FFTS scheduler raises `coredim=X can't be greater than UINT16_MAX` if any
grid dimension exceeds 65535. Loop over N and launch sub-grids of `C*H` when `N*C*H > 65535`.

### Step 3: Shape table

Cover **all dispatch paths** of the optimized kernel — not just the benchmark shape.

```python
_BENCH_SHAPES = [
    # label              N    C   H_out  W_out
    ("N1-C256-14x14",    1,  256,   14,    14),  # HW=196   small → persistent path
    ("N1-C128-28x28",    1,  128,   28,    28),  # HW=784   small → persistent path
    ("N8-C64-14x14",     8,   64,   14,    14),  # N>1 small
    ("N1-C64-56x56",     1,   64,   56,    56),  # HW=3136  large → loop path
    ("N1-C96-56x56",     1,   96,   56,    56),  # non-pow2 C
    ("N1-C32-224x224",   1,   32,  224,   224),  # HW=50176 large, many tiles
    ("N128-C128-126x126",128, 128,  126,   126),  # benchmark shape
]
```

Include at minimum:
- One small-HW shape (tests the persistent/startup-amortised path)
- One large-HW shape (tests the tiled loop path)
- One non-power-of-2 dimension (tests that `pid % C` is safe)
- The exact benchmark shape from the bench file

### Step 4: `@triton.testing.perf_report` — always use this decorator

**Never** hand-roll a benchmark loop:

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
    x    = torch.rand(N, C, H_out, W_out, device="npu", dtype=torch.float16)
    bias = torch.rand(C, 1, 1, device="npu", dtype=torch.float16)
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
axis labelling using the `ylabel` string — just return the raw seconds value.

**Do NOT use hex colour codes or composite linestyles.** The `perf_report` style parser only accepts:
- Plain named colours: `"blue"`, `"red"`, `"green"`, `"orange"`, etc.
- Simple linestyles: `"-"`, `"--"`, `"-."`, `":"`
- NOT accepted: `"#4C72B0"` (hex), `"-o"` (linestyle+marker combined)

Correct: `styles=[("blue", "-"), ("red", "-"), ("green", "-")]`

### Step 5: Unit test

Always include a correctness check:

```python
def unit_test():
    torch.manual_seed(42)
    any_fail = False
    for label, N, C, H_out, W_out in _BENCH_SHAPES:
        x         = torch.rand(N, C, H_out, W_out, device="npu", dtype=torch.float16) * 4 - 2
        bias      = torch.rand(C, 1, 1, device="npu", dtype=torch.float16) * 0.5
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

### Step 6: `main()` and entry point

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

## Complete File Template

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

# ── runner functions ───────────────────────────────────────────────────────────

def _run_torch_ref(x, bias):
    return torch.nn.functional.relu(x) + bias

def _run_baseline(x, bias_flat):
    N, C, H, W = x.shape
    y       = torch.empty_like(x)
    block_w = 128 if W >= 128 else (64 if W >= 64 else 32)
    w_grid  = triton.cdiv(W, block_w)
    if N * C * H <= 65535:
        _baseline._kernel[(N * C * H, w_grid)](
            x, y, bias_flat, N, C, H, W, BLOCK_W=block_w, num_warps=4)
    else:
        for n in range(N):
            _baseline._kernel[(C * H, w_grid)](
                x[n:n+1], y[n:n+1], bias_flat, 1, C, H, W,
                BLOCK_W=block_w, num_warps=4)
    return y

def _run_optimized(x, bias):
    return _optimized._dispatch_fn(x, bias)

# ── shapes ─────────────────────────────────────────────────────────────────────

_BENCH_SHAPES = [
    ("N1-C256-14x14",    1, 256,  14,  14),
    ("N1-C128-28x28",    1, 128,  28,  28),
    ("N8-C64-14x14",     8,  64,  14,  14),
    ("N1-C64-56x56",     1,  64,  56,  56),
    ("N1-C96-56x56",     1,  96,  56,  56),
    ("N1-C32-224x224",   1,  32, 224, 224),
    ("N128-C128-126x126",128,128, 126, 126),
]

# ── benchmark ──────────────────────────────────────────────────────────────────

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
    bias      = torch.rand(C, 1, 1, device="npu", dtype=torch.float16)
    bias_flat = bias.reshape(-1)
    fn = (_run_torch_ref  if mode == "torch_ref"  else
          _run_baseline   if mode == "baseline"   else _run_optimized)
    call = (lambda: fn(x, bias)) if mode != "baseline" else (lambda: fn(x, bias_flat))
    return triton.testing.do_bench(call, warmup=25, rep=200, return_mode="mean")

# ── unit test ──────────────────────────────────────────────────────────────────

def unit_test():
    torch.manual_seed(42)
    any_fail = False
    for label, N, C, H, W in _BENCH_SHAPES:
        x         = torch.rand(N, C, H, W, device="npu", dtype=torch.float16) * 4 - 2
        bias      = torch.rand(C, 1, 1, device="npu", dtype=torch.float16) * 0.5
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

# ── entry point ────────────────────────────────────────────────────────────────

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

- **Wrong kernel args** — always read the `@triton.jit` signature directly from the source file
- **Grid overflow `coredim > UINT16_MAX`** — guard: if `N*C*H > 65535`, loop over N with sub-grids of `C*H`
- **`* 1e3` scaling** — `do_bench` returns seconds; the table will show values like `0.045` ms
- **Style format** — use plain named colours (`"blue"`, `"red"`, `"green"`) and simple linestyles (`"-"`, `"--"`, `"-."`)
- **Shapes must cover ALL dispatch paths** — include small-HW, large-HW, non-power-of-2, and benchmark shape
- **Load via importlib, never copy-paste** — if you inline kernel code into `profile_kernels.py`, the profile diverges from the file on disk

## Constraints
- Generate this script AFTER optimized kernel is validated via cannsim
- This script runs on real NPU hardware, not cannsim
- Always use `@triton.testing.perf_report` — never hand-roll benchmark loops
- Return `triton.testing.do_bench(fn, warmup=25, rep=200, return_mode="mean")` directly (no scaling)
