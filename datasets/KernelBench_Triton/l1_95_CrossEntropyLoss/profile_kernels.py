import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import torch_npu  # noqa: F401
import triton

_DIR = Path(__file__).parent


# ── kernel loading ─────────────────────────────────────────────────────────────

def _load(fname):
    spec = importlib.util.spec_from_file_location(fname.stem, fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_baseline_mod1 = _load(_DIR / "95_CrossEntropyLoss.py")
_baseline_mod2 = _load(_DIR / "base_95_CrossEntropyLoss.py")
_optimized_mod = _load(_DIR / "opt_95_CrossEntropyLoss.py")


# ── runner functions ───────────────────────────────────────────────────────────

def _run_torch_ref(x, t):
    return torch.nn.functional.cross_entropy(x, t, reduction='none')

_baseline_model1 = _baseline_mod1.ModelNew()

def _run_baseline1(x, t):

    return _baseline_model1(x, t)


_baseline_model2 = _baseline_mod2.ModelNew()

def _run_baseline2(x, t):

    return _baseline_model2(x, t)

_optimized_model = _optimized_mod.ModelNew()

def _run_optimized(x, t):

    return _optimized_model(x, t)


# ── shapes ─────────────────────────────────────────────────────────────────────
_BENCH_SHAPES = [
    ("M256-N256",    256,   256),   
    ("M512-N512",   512,  512),  
    ("M512-N1024",   512,  1024),  
    ("M1024-N2048",   1024,  2048), 
    ("M1024-N4096",   1024,  4096),  
    ("M4000-N4000", 4000,  4000),   
    ("M4000-N4000", 4000,  4000),   
    ("M2**14-N2**14", 2**14,  2**14),       
]


# ── benchmark ──────────────────────────────────────────────────────────────────

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["label"],
        x_vals=[s[0] for s in _BENCH_SHAPES],
        line_arg="mode",
        line_vals=["torch_ref", "baseline1", "baseline2", "optimized"],
        line_names=["PyTorch / ACL", "Baseline Triton1", "Baseline Triton2", "Optimized Triton"],
        styles=[("blue", "-"), ("red", "-"), ("black", "-"), ("green", "-")],
        ylabel="Latency (ms)",
        plot_name="crossentropy_perf",
        args={},
    )
)
def benchmark(label, mode):
    _, M, N = next(s for s in _BENCH_SHAPES if s[0] == label)
    x = torch.rand(M, N, device="npu", dtype=torch.float32)
    target = torch.randint(low=0, high=N, size=(M,), device='npu', dtype=torch.int64)

    if mode == "torch_ref":
        fn = lambda: _run_torch_ref(x, target)
    elif mode == "baseline1":
        fn = lambda: _run_baseline1(x, target)
    elif mode == "baseline2":
        fn = lambda: _run_baseline2(x, target)
    else:
        fn = lambda: _run_optimized(x, target)

    # do_bench returns seconds; perf_report ylabel is "Latency (ms)"
    return triton.testing.do_bench(fn, warmup=25, rep=200, return_mode="mean")


# ── unit test ──────────────────────────────────────────────────────────────────

def unit_test():
    torch.manual_seed(42)
    any_fail = False

    for label, n_samples, n_classses in _BENCH_SHAPES:
        # Use values in range that produce clearly nonzero cumulative sums
        x = torch.rand(n_samples, n_classses, device="npu", dtype=torch.float32)
        target = torch.randint(low=0, high=n_classses, size=(n_samples,), device='npu', dtype=torch.int64)

        ref  = _run_torch_ref(x.clone(), target.clone())
        base1 = _run_baseline1(x.clone(), target.clone())
        base2 = _run_baseline2(x.clone(), target.clone())
        opt  = _run_optimized(x.clone(), target.clone())

        ok_b1 = torch.allclose(ref, base1, atol=1e-2, rtol=1e-2)
        ok_b2 = torch.allclose(ref, base2, atol=1e-2, rtol=1e-2)
        ok_o = torch.allclose(ref, opt,  atol=1e-2, rtol=1e-2)

        maxdelta_b1 = (ref - base1).abs().max().item()
        maxdelta_b2 = (ref - base2).abs().max().item()
        maxdelta_o = (ref - opt).abs().max().item()

        print(f"  {label:<14}  "
              f"baseline1 [{('PASS' if ok_b1 else 'FAIL')}]  "
              f"baseline2 [{('PASS' if ok_b2 else 'FAIL')}]  "
              f"optimized [{('PASS' if ok_o else 'FAIL')}]  "
              f"maxΔ_base1={maxdelta_b1:.2e}  "
              f"maxΔ_base2={maxdelta_b2:.2e}  "
              f"maxΔ_opt={maxdelta_o:.2e}")

        if not ok_b1 or not ok_b2 or not ok_o:
            any_fail = True

    if any_fail:
        raise AssertionError("Correctness check failed — see [FAIL] lines above")
    print("All PASS")


# ── entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Cumsum kernel benchmark")
    parser.add_argument("--test",  action="store_true", help="Correctness check only")
    parser.add_argument("--bench", action="store_true", help="Benchmark only")
    args = parser.parse_args()

    run_test  = args.test  or not args.bench
    run_bench = args.bench or not args.test

    if run_test:
        print("=== Unit test ===")
        unit_test()

    if run_bench:
        print("=== Benchmark ===")
        benchmark.run(save_path=str(_DIR), print_data=True)
        # → prints table + saves cumsum_perf.png automatically


if __name__ == "__main__":
    main()
