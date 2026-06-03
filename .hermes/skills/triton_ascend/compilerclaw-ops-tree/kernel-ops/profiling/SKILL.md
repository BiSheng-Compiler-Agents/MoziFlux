---
name: profiling
description: "Write and run `profile_kernels.py` for Triton kernels on Ascend NPU hardware. Covers multi-line comparison (torch_ref vs 1-2 baselines vs optimized), weight-init matching between torch_ref and baseline models, lazy NPU model init, and `@perf_report` benchmark decorator usage."
---

# Kernel Hardware Profiling [LEAF NODE]

Write and run `profile_kernels.py` for Triton kernels on Ascend NPU hardware.
Covers the full workflow: script structure, `@perf_report` usage, multi-line
comparison (torch_ref vs baselines vs optimized), correctness test with
matching weight initialization, lazy NPU model init, and grid overflow guard.
Runs on a **real Ascend NPU** (not cannsim) — for final wall-clock latency measurement.

## When to generate this script

Generate `profile_kernels.py` **after** the optimized kernel is written and its correctness
has been validated via cannsim. It belongs alongside the kernel files:

```
l2_<N>_<KernelName>/
├── <N>_<KernelName>.py              ← baseline1 Triton kernel (e.g. fast/specific path)
├── base_<N>_<KernelName>.py         ← baseline2 Triton kernel (e.g. generic path, optional)
├── opt_<N>_<KernelName>.py          ← optimized kernel (all shapes)
└── profile_kernels.py               ← this script (generated last)
```

Some kernels only have one baseline (no `base_*`), in which case use the 3-line format.

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

_baseline1  = _load(_DIR / "<N>_<KernelName>.py")
_baseline2  = _load(_DIR / "base_<N>_<KernelName>.py")   # optional, if exists
_optimized = _load(_DIR / "opt_<N>_<KernelName>.py")
```

### Step 2: Lazily-instantiated Triton models

**IMPORTANT:** Do NOT instantiate `ModelNew(...)` at module import time — those
objects are created on CPU. If `x` lives on NPU, passing it through CPU-resident
parameters triggers `RuntimeError: expected all tensors to be on one device`.

Use a lazy-init cache instead:

```python
_baseline1_model = None
_baseline2_model = None   # optional
_optimized_model = None

def _ensure_models_on_npu():
    """Lazily create models on NPU on first call (avoids CPU/NPU mismatch)."""
    global _baseline1_model, _baseline2_model, _optimized_model
    if _baseline1_model is None:
        torch.manual_seed(0)
        _baseline1_model = _baseline1.ModelNew(
            *_baseline1.get_init_inputs()).to(device="npu", dtype=torch.float32).eval()
        # If second baseline exists:
        torch.manual_seed(0)
        _baseline2_model = _baseline2.ModelNew(
            *_baseline2.get_init_inputs()).to(device="npu", dtype=torch.float32).eval()
        torch.manual_seed(0)
        _optimized_model = _optimized.ModelNew(
            *_optimized.get_init_inputs()).to(device="npu", dtype=torch.float32).eval()
```

### Step 3: Runner functions

**Always** go through `_run_*` wrappers — both in benchmark AND unit_test. Never
call `_baseline1_model(x)` directly outside a runner function. The runners call
`_ensure_models_on_npu()` internally, so the lazy init stays in one place.

```python
_torch_ref_cache: dict = {}

def _run_torch_ref(x: torch.Tensor) -> torch.Tensor:
    """Pure PyTorch pipeline — MUST match the Triton kernel's math exactly."""
    device_key = getattr(x.device, "index", None) or 0
    cached = _torch_ref_cache.get(device_key)
    if cached is None:
        # Build the PyTorch ops lazily, seeded to match baseline weight init.
        # The torch_ref Conv3d + bias must share torch.manual_seed(0) with the
        # baseline ModelNew so that nn.Conv3d default-init weights match exactly.
        torch.manual_seed(0)
        conv = nn.Conv3d(in_ch, out_ch, kernel_size).to(device=x.device, dtype=torch.float32)
        # ... MaxPool, AdaptiveAvgPool, bias Parameter ...
        _torch_ref_cache[device_key] = (conv, max_pool, avg_pool, bias)
    conv, max_pool, avg_pool, bias = _torch_ref_cache[device_key]
    with torch.no_grad():
        w, b = conv.weight, conv.bias
        x = F.conv3d(x, w / divisor, b / divisor, ...)
        x = max_pool(x)
        x = avg_pool(x)
        # ... reshape, add bias, sum/reduce ...
    return out

def _run_baseline1(x):
    _ensure_models_on_npu()
    return _baseline1_model(x)

def _run_baseline2(x):              # optional, only if base_* exists
    _ensure_models_on_npu()
    return _baseline2_model(x)

def _run_optimized(x):
    _ensure_models_on_npu()
    return _optimized_model(x)
```

**For kernels that do a multi-op pipeline (Conv3d -> divide -> pool -> reduce),
the torch_ref is NOT just a built-in op — it must replicate the full pipeline with
`F.conv3d`, `nn.MaxPool3d`, `nn.AdaptiveAvgPool3d`, etc.**

**Always add the 65535 grid overflow guard for the baseline when calling Triton
kernels directly (not needed when going through a `.ModelNew` wrapper).**
Ascend's FFTS scheduler raises `coredim=X can't be greater than UINT16_MAX`
if any grid dimension exceeds 65535.

### Step 4: Shape table

Cover **all dispatch paths** — not just the benchmark shape. Use the SAME
`_BENCH_SHAPES` list for both benchmark and unit_test.

```python
_BENCH_SHAPES = [
    # label              B    (or N, C, H, W as appropriate for the kernel)
    ("B=1",              1),
    ("B=4",              4),
    ("B=16",            16),
    ("B=32",            32),
    ("B=64",            64),
    ("B=128",          128),
    ("B=256",          256),
]
```

Include at minimum:
- The exact benchmark shape from results.txt
- If varied: small & large sizes, non-power-of-2 dimensions

### Step 5: Canonical benchmark column names — HARD RULE

The first entry in `line_names` list inside the `Benchmark(...)` constructor is the **PyTorch/ACL reference implementation** and its display name is fixed across the whole project:

```
line_names[0] MUST be exactly  "PyTorch / ACL"
```

**Forbidden display names for the reference slot**:

| Wrong (do not use)               | Why                                            |
|----------------------------------|------------------------------------------------|
| `"torch_npu"`                    | Backend name, not the user-facing method name |
| `"torch.matmul"` / `"torch.add"` | API call, not a benchmark column label        |
| `"PyTorch cumsum"` / `"PyTorch conv2d"` | Operation-specific; not a generic column name |
| `"PyTorch ref"`                  | Ambiguous; collapses PyTorch and ACL          |
| `"Reference"` / `"Ref"`          | Too short; doesn't say PyTorch or ACL          |
| `"acl"` / `"ACL"`                | Backend only; missing PyTorch                  |
| `"baseline"`                     | That's the Triton baseline slot                |

Fixed display names for Triton slots (extend as needed for multiple baselines):

```
line_names[1] = "Baseline Triton1"     # <N>_<KernelName>.py
line_names[2] = "Baseline Triton2"     # base_<N>_<KernelName>.py (if exists)
line_names[3] = "Optimized Triton"     # opt_<N>_<KernelName>.py
```

If only one baseline exists, use 3-line format:
```
line_names[1] = "Baseline Triton"
line_names[2] = "Optimized Triton"
```

The internal `line_vals` discriminator may use any short token you like.

### Step 6: `@triton.testing.perf_report` — always use this decorator

**Never** hand-roll a benchmark loop:

**4-line format** (when two baselines exist):
```python
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["label"],
        x_vals=[s[0] for s in _BENCH_SHAPES],
        line_arg="mode",
        line_vals=["torch_ref", "baseline1", "baseline2", "optimized"],
        line_names=["PyTorch / ACL", "Baseline Triton1", "Baseline Triton2", "Optimized Triton"],
        styles=[("blue", "-"), ("red", "-"), ("black", "-"), ("green", "-")],
        ylabel="Latency (ms)",
        plot_name="<kernel_name>_perf",
        args={},
    )
)
```

**3-line format** (single baseline):
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
    _, N, C, H, W = next(s for s in _BENCH_SHAPES if s[0] == label)
    x = torch.rand(N, C, H, W, device="npu", dtype=torch.float16)

    if mode == "torch_ref":
        fn = lambda: _run_torch_ref(x)
    elif mode == "baseline":
        fn = lambda: _run_baseline(x)
    else:
        fn = lambda: _run_optimized(x)

    # do_bench returns seconds — perf_report labels y-axis as the ylabel above
    return triton.testing.do_bench(fn, warmup=25, rep=200, return_mode="mean")
```

**Do NOT multiply by 1e3.** `do_bench` returns seconds. `perf_report` handles
axis labelling using the `ylabel` string — just return the raw seconds value.

**Do NOT use hex colour codes or composite linestyles.** The `perf_report` style parser only accepts:
- Plain named colours: `"blue"`, `"red"`, `"green"`, `"black"`, `"orange"`, etc.
- Simple linestyles: `"-"`, `"--"`, `"-."`, `":"`
- NOT accepted: `"#4C72B0"` (hex), `"-o"` (linestyle+marker combined)

### Step 7: Unit test

Always include a correctness check. **Reuse `_run_torch_ref()`** — do NOT
rebuild the PyTorch pipeline inline in unit_test. This guarantees the reference
logic is defined in exactly one place.

```python
def unit_test():
    print("=" * 60)
    print("UNIT TEST — correctness vs torch_ref")
    print("=" * 60)
    torch.manual_seed(42)
    any_fail = False

    # Pre-warm torch_ref cache with matching seed (same seed used by baselines)
    torch.manual_seed(0)
    _run_torch_ref(torch.rand(1, in_ch, D, H, W, device="npu", dtype=torch.float32))

    for label, B in _BENCH_SHAPES:
        torch.manual_seed(42)
        x = torch.rand(B, in_ch, D, H, W, device="npu", dtype=torch.float32)
        with torch.no_grad():
            ref  = _run_torch_ref(x)
            base1 = _run_baseline1(x.clone())
            base2 = _run_baseline2(x.clone())   # optional
            opt   = _run_optimized(x.clone())

        # ... compare with torch.allclose, print PASS/FAIL ...
```

**Critical: weight-init matching.** When the Triton baseline's `ModelNew.__init__`
constructs `nn.Conv3d(...)` + `nn.Parameter(torch.randn(bias_shape))`, those
consume from `torch.manual_seed(0)`. The torch_ref must replicate the EXACT
same construction order with the SAME seed so weights and biases match. The
easiest way: let `_run_torch_ref()` lazy-init with `torch.manual_seed(0)` inside
the cache miss path, then call it once before the test loop to pre-warm.

### Step 8: `main()` and entry point

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

if __name__ == "__main__":
    main()
```

---

## Pitfalls

- **Wrong kernel args** — always read the `@triton.jit` signature directly from the source file
- **Grid overflow `coredim > UINT16_MAX`** — guard: if `N*C*H > 65535`, loop over N with sub-grids of `C*H`
- **`* 1e3` scaling** — `do_bench` returns seconds; the table will show values like `0.045` ms
- **Style format** — use plain named colours (`"blue"`, `"red"`, `"green"`, `"black"`) and simple linestyles (`"-"`, `"--"`, `"-."`)
- **Shapes must cover benchmark shape** — and both dispatch paths if the kernel has them
- **Load via importlib, never copy-paste** — if you inline kernel code into `profile_kernels.py`, the profile diverges from the file on disk
- **Reference column name is FIXED to `"PyTorch / ACL"`** — see Step 5. If you inherit a profile with a wrong name, fix it.
- **Device mismatch at import time** — module-level `ModelNew(...)` is CPU. Models that will be called with NPU input MUST be lazily moved to NPU (see Step 2). Error: `expected all tensors to be on one device but found cpu and npu`.
- **Access model globals directly** — always go through `_run_*` wrappers, never `_baseline1_model(x)` at module scope or inside unit_test loops. The `_run_*` functions own the lazy-init call.
- **Duplicated torch_ref logic in unit_test** — call `_run_torch_ref()`, don't rebuild. One source of truth.

## `line_names` for multi-baseline kernels

| Slots                    | line_vals                              | line_names                                              |
|--------------------------|----------------------------------------|---------------------------------------------------------|
| 3-line (1 baseline)      | `["torch_ref", "baseline", "optimized"]` | `["PyTorch / ACL", "Baseline Triton", "Optimized Triton"]` |
| 4-line (2 baselines)     | `["torch_ref", "baseline1", "baseline2", "optimized"]` | `["PyTorch / ACL", "Baseline Triton1", "Baseline Triton2", "Optimized Triton"]` |

## Reference template

A complete, working `profile_kernels.py` is checked in at
`references/profile_kernels.py`. The l2_1_Conv2D_ReLU_BiasAdd version is the
reference for the 4-line (two baseline) format.

## Constraints
- Generate this script AFTER optimized kernel is validated via cannsim
- This script runs on real NPU hardware, not cannsim
- Always use `@triton.testing.perf_report` — never hand-roll benchmark loops
- Return `triton.testing.do_bench(fn, warmup=25, rep=200, return_mode="mean")` directly (no scaling)
