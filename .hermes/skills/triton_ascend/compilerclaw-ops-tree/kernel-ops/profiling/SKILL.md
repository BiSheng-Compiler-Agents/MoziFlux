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

## Remote Execution

This script is designed to run on a remote Ascend machine via the `remote_verify` tool.
The agent should NOT run profile_kernels.py locally — it requires a physical NPU.

Workflow:
1. Generate `profile_kernels.py` in the workspace directory
2. Call `remote_verify(local_dir=workspace, run_test=True, run_bench=True)`
3. The tool uploads all files, runs correctness then benchmark, downloads results
4. Analyze results from `workspace/remote_results/` directory

Required env vars for remote execution:
```
REMOTE_VERIFY_HOST=192.168.1.10
REMOTE_VERIFY_USER=username
REMOTE_VERIFY_PASS=password
```

### CANN Environment on Remote (Critical)

The remote machine must source the CANN toolkit environment **before** running
`torch_npu`. Without it: `ImportError: libhccl.so` or
`RuntimeError: Failed to load the backend extension: torch_npu`.

The `remote_verify` plugin (v1.0.1+) handles this automatically. If NOT using
`remote_verify`, source manually:
```bash
source ~/miniconda3/envs/compilerclaw/Ascend/cann-9.0.0/set_env.sh
export TRITON_NPU_COMPILER_PATH=.../Ascend/cann-9.0.0/x86_64-linux/bin/bisheng
conda run -n compilerclaw python profile_kernels.py --test
```

See `references/remote_execution_lessons.md` for pitfalls: per-variant
resilience, correctness checking every variant against torch_ref, upload exclusion,
and the `if __name__ == "__main__"` entry-point requirement.

## When to Generate

Generate `profile_kernels.py` **after** the optimized kernel is written and its correctness
has been validated via cannsim. It belongs alongside the kernel files:

```
l2_<N>_<KernelName>/
├── <N>_<KernelName>.py              ← baseline1 Triton kernel
├── base_<N>_<KernelName>.py         ← baseline2 Triton kernel (optional)
├── opt_<N>_<KernelName>.py          ← optimized kernel
└── profile_kernels.py               ← this script (generated last)
```

## Script Structure

Use `templates/profile_kernels.py` as the canonical template.
Adapt it for your kernel — change file names, shapes, input construction, and torch_ref logic.

Key structural requirements (all enforced in the template):

1. **Load via `importlib`** — never copy-paste kernel code. Use the `_load()` helper with `k_` prefix.
2. **Lazy NPU model init** — do NOT instantiate `ModelNew(...)` at module level. Use `_model(key)` cache.
3. **Runner wrappers** — `_run_torch_ref()`, `_run_provider(key, x)`. Never call models directly.
4. **Shape table** — `_BENCH_SHAPES` with `(label, dims...)` tuples. Labels MUST contain NO spaces. Shapes must match the kernel's stated requirement/name and the original `get_inputs()` contract; do not benchmark unrelated regimes.
   - `large_K` / "large K dimension" kernels: every benchmark shape must keep K large relative to M/N, plus include the exact required benchmark shape.
   - `small_K` / "small K dimension" kernels: every benchmark shape must keep K small relative to M/N.
   - `tall_skinny` kernels: preserve the original skinny dimension from `get_inputs()`. If `get_inputs()` is `A=(M,K), B=(K,M)` with `M >> K`, keep K small and N=M; do not accidentally make both output dimensions huge without guarding grid limits.
   - irregular-shape kernels: include non-powers-of-two and boundary shapes, not only square powers of two.
   - Before launching any Triton provider, compute the most conservative possible launch grid from the smallest autotune block sizes; if it exceeds `coreDim <= 65535`, pre-skip that provider/shape and return `inf` rather than poisoning the NPU context.
5. **All baselines** — include every existing provider: PyTorch / ACL, `<N>_*.py` as `Baseline Triton1`, `base_<N>_*.py` as `Baseline Triton2` when present, and `opt_<N>_*.py` as `Optimized Triton`. Do not silently drop `base_*.py`.
6. **Correctness coverage is gating** — run correctness for EVERY provider on EVERY shape in `_BENCH_SHAPES`; benchmark-only shapes are forbidden unless explicitly skipped with a printed reason. Benchmark numbers are meaningful only for providers that produce the same outputs as `PyTorch / ACL`. If an editable provider fails correctness, investigate and fix the root cause before trusting or comparing its benchmark numbers. Only if a comparison provider cannot be fixed or is intentionally read-only should it remain in the table with `inf`/skip timing. Gate `UNIT_TEST PASS/FAIL` on optimized correctness and reference construction, but do not hide broken comparison baselines behind `INFO`: fix them. After any fix, run `remote_verify` yourself and require real `test_passed=true` + `bench_passed=true` before saying the profiler is correct.
   - For very large Ascend NPU tensors, `torch.testing.assert_close` can emit misleading mismatch counts (especially with internal-format tensors) even when `max_abs_diff` is within tolerance. In `profile_kernels.py`, compute a scalar max absolute diff and gate on the dtype tolerance when `assert_close` behaves pathologically; still print the max diff for every provider/shape.
7. **Benchmark** — use `@triton.testing.perf_report` with `x_names=["label"]` for parser-friendly tables. Inside each benchmark cell, prefer `torch_npu.profiler` device timing when comparing NPU kernels or ACL fused ops; parse `op_statistic.csv` `Avg Time(us)` first and fall back to `kernel_details.csv` `Duration(us)`. Use manual `time.perf_counter` + `torch.npu.synchronize()` only when profiler output is unavailable or too costly. See `references/torch-npu-profiler.md`. For ops where `do_bench` timing is unreliable or you need device-level kernel durations, use `torch_npu.profiler` instead — see `references/torch-npu-profiler.md` for API details and the sync pattern.
8. **Resilience / NPU context poisoning** — use the standard single-process profiling template; do **not** wrap providers in subprocesses. Wrap provider calls in try/except and return `float("inf")` on benchmark failure. Guard launches known to poison the current process before executing them: Ascend runtime errors such as `coreDim > 65535` can make later providers fail or report bogus timings. If a comparison provider passes unit correctness but poisons the NPU context during benchmark warmup/timing, keep its correctness test and visible benchmark column, but pre-skip only its timing cells with `INFO ... inf comparison_provider_preskipped_to_avoid_npu_context_poisoning`; do not let it run before optimized timing. If a provider is known to MLIR-abort for all shapes, keep the column and return `inf` for that provider with `INFO` wording, not `ERROR`/`FAIL`, so parser-driven verification can still pass when optimized correctness passes. Do not use subprocess isolation to make aborts catchable unless the user explicitly asks. If the user says a provider is editable and should be comparable, find and fix the provider root cause instead of masking it as `INFO`/`inf`.
9. **Input construction must match the source kernel** — derive tensors from the original `<N>_*.py` `get_inputs()` contract: shape, dtype, layout/contiguity, and distribution (`torch.rand` vs `torch.randn`). Do not use fp32 test tensors when the source kernel only accepts fp16/bf16; do not change random distribution when comparing correctness.
10. **Fix editable provider correctness before trusting timing** — if `Baseline Triton1` (`<N>_*.py`) or another editable comparison provider produces NaNs/large errors, find and fix the root cause before accepting benchmark numbers. Common Ascend matmul cause: fp16/bf16 operands were upcast with `.to(tl.float32)` before `tl.dot`; keep operands native and use an fp32 accumulator (`acc += tl.dot(a, b)` or `tl.dot(a, b, acc)`).
11. **No subprocess wrappers** — `profile_kernels.py` must follow the canonical profiling template in a single Python process with importlib-loaded providers, per-provider try/except cells, and `perf_report`. Do not wrap each provider/shape in subprocesses unless the user explicitly asks; subprocess isolation breaks the template/parser expectations and hides normal profiler structure.
12. **Entry point** — `if __name__ == "__main__": main()` block is REQUIRED. Keep process exit code zero even when correctness fails so benchmark output/results artifacts are still produced and downloadable; print `UNIT_TEST_FAILED` instead of `sys.exit(1)`.

## Benchmark Column Names — HARD RULE

```
line_names[0] = "PyTorch / ACL"          # reference (fixed, never change)
line_names[1] = "Baseline Triton1"       # <N>_<KernelName>.py
line_names[2] = "Baseline Triton2"       # base_<N>_<KernelName>.py (if exists)
line_names[3] = "Optimized Triton"       # opt_<N>_<KernelName>.py
```

For single-baseline (3-line): `"Baseline Triton"` instead of `"Baseline Triton1"`.

Forbidden reference names: `"torch_npu"`, `"torch.matmul"`, `"PyTorch ref"`, `"Reference"`, `"Ref"`, `"acl"`, `"ACL"`, `"baseline"`.

## Parser Contract

`results.txt` is consumed by `generate_report.py`. A profile that runs fine on hardware
can still produce **0 parsed records** if the format is wrong. See
`references/generate_report_parser_contract.md` for the 5 hard requirements:

1. `x_names=["label"]` only (no multi-column shapes)
2. Header first token must be `label` or `N`
3. Method names must match the alias table exactly
4. Geomean speedup needs both `Baseline Triton1` AND `Optimized Triton`
5. Label strings must contain NO spaces

Quick self-check: run captured output through `generate_report.py` and confirm `recs > 0`.

## Benchmark via torch_npu.profiler

The canonical template (`templates/profile_kernels.py`) benchmarks with `torch_npu.profiler`,
not `do_bench`. Rules learned on hardware:

- Wrap each provider call inside a `with profile(...)` context in `benchmark(label, mode)`;
  synchronize (`torch.npu.synchronize()`) **before the loop and after every iteration**, then
  `prof.step()`. Missing per-iteration sync inflated measured durations ~10x (140 ms → 1401 ms).
- Use `schedule(wait=1, warmup=2, active=5, repeat=1)` and parse
  `ASCEND_PROFILER_OUTPUT/op_statistic.csv` (`Avg Time(us)`) first; fall back to
  `kernel_details.csv` (`Duration(us)`). See `references/torch-npu-profiler.md` for
  `_ExperimentalConfig`, levels, metrics, and output files.
- One profiler context per (provider, shape) cell, each with a fresh tempdir; delete it after
  parsing unless `KEEP_FA_PROF=1` (or equivalent) is set.
- No subprocess isolation by default — profile in-process in `benchmark()`. Do not wrap cells in
  subprocesses unless the user explicitly asks.
- Template TEST lines use canonical lowercase keys `baseline1`, `baseline2`, `optimized`
  (required by kernel-sandbox `_validate_results_txt`), `UNIT_TEST PASS` gates on optimized
  correctness, and the script always exits 0 so results download.
- `do_bench` remains only a fallback when profiler is unavailable.

## Pitfalls

- **Instruction-only requests** — when the user asks how to patch/debug a package or asks for the command/snippet, provide instructions only unless they explicitly ask to edit the environment. Do not modify site-packages or project files while answering explanatory Q&A; if you accidentally do, revert only that change.
- **Profiler timing with `torch_npu.profiler`** — put only the op being measured inside the profiler scope, run it in a loop for `wait + warmup + active` steps, call `torch.npu.synchronize()` before the loop and after every op before `prof.step()`, and use `tensorboard_trace_handler(..., analyse_flag=True)` so CSV files are generated. `Level0` gives low-overhead timing; `Level1/2` add AIC metrics and can change overhead.
- **Remote verify disk-full masquerading as FileNotFoundError** — if `remote_verify` suddenly returns `[Errno 2] No such file` before remote stdout, diagnose remote directory creation/upload. A full remote filesystem can make `mkdir` fail and SFTP then reports `FileNotFoundError`; check `df -h` and clean old profiler/job outputs before retrying.
- **Activation reference fallback** — when a native PyTorch/ACL activation fails during reference construction on a CANN/torch_npu build, replace only the reference expression with an exact algebraic equivalent and keep optimized correctness gated against it. Example: for `ReLU(HardSwish(x))`, use `torch.where(x > 0, x * torch.clamp(x + 3, max=6) / 6, 0)` instead of `F.relu(F.hardswish(x))` if native `hardswish` fails. See `references/activation-reference-fallback.md`.
- **Wrong kernel args** — always read the `@triton.jit` signature directly from the source file; also read the host-side entry function's signature. When swapping which file `_load()`s as a provider, adapt the `_run_provider()` call to that file's entry convention (e.g. one FA variant's `attention` takes `(q, k, v, atten_mask, causal, sm_scale, BM, BN)`, not the template's `(q, k, v, sm_scale, causal)`) — a stale provider call fails with TypeError before the kernel ever compiles.
- **Grid overflow / `coreDim > 65535`** — guard before launching the invalid variant in both unit tests and benchmarks; a failed Ascend launch can poison the NPU context and make later providers return bogus errors/timings. For fixable kernels, loop over the overflowing dimension with sub-grids.
- **MLIR abort / invalid-provider handling** — Python try/except cannot catch an abort from MLIR. If investigation shows a provider aborts for every shape, do not use subprocess isolation in `profile_kernels.py`; keep the provider column and return `inf` for that provider with a clear printed reason. For expected/handled provider-wide `inf`, avoid printing `ERROR`/`FAIL` in unit-test output if the verifier treats those tokens as failure; use `INFO`/`SKIP`/`INF` wording and keep exit code zero. **Do not interpolate raw exception strings for comparison providers**: MLIR/Triton exceptions often contain `[ERROR]`/`[FAIL]` tokens internally, which can make parser-driven verification fail even when optimized correctness passes. Print only the exception type or a sanitized one-line reason, e.g. `INFO comparison_provider_unavailable Baseline Triton2 shape: MLIRCompilationError`. Never hide correctness failures for an editable provider: if baseline/input code is part of the deliverable and is wrong, find and fix the root cause before trusting benchmark numbers.
- **Autotune grid mismatch** — if the kernel uses `@triton.autotune`, the host `grid` must be derived from the selected `META` when possible. A conservative min-block grid is correctness-safe but may overlaunch many extra program instances and make the optimized path slower than a grouped baseline; inspect `optimization/references/grid-autotune-mismatch.md` before accepting those timings.
- **`do_bench` returns miliseconds** — do NOT multiply by 1e3; `perf_report` handles axis labelling
- **`do_bench` + `@triton.autotune` warmup** — `do_bench` warms up the kernel by running it `warmup` times before timing. During warmup, autotune runs all configs and selects the fastest; only the winning config is timed. So autotune overhead is paid during warmup and is NOT included in the reported latency. However, if `do_bench` fails or is unavailable, the `time.perf_counter` fallback must manually implement warmup+rep.
- **Autotune key params must match kernel signature** — if `@triton.autotune(key=['n_elements_pow2'])` is used, EVERY kernel decorated by it (including persistent kernels) must declare `n_elements_pow2: tl.constexpr` in its signature. Missing constexpr param causes `RuntimeError: No valid triton configs. NoneType: None` on large shapes that trigger the affected dispatch path.
- **`get_init_inputs()` returning `[()]`** — some kernels return `[()]` (a list with one empty tuple) from `get_init_inputs()`, meaning "no constructor args". The profile's `_model()` must detect this and treat it as `[]`, otherwise `ModelNew(*[()])` passes the empty tuple as a positional arg and raises `TypeError: ModelNew.__init__() takes 1 positional argument but 2 were given`. Fix pattern: `if init == [()]: init = []` before calling `ModelNew(*init)`.
- **Persistent kernel tile loop** — in a persistent (grid-capped) kernel, the work loop must iterate over **tiles**, not elements. Use `n_tiles = tl.cdiv(n_elements, BLOCK_SIZE)` then `for tile_id in range(pid, n_tiles, n_programs)`. Using `n_elements` instead of `n_tiles` in the range causes each program to skip by `n_elements` instead of processing consecutive tiles, producing wrong output or OOB access.
- **base_*.py pre-existing bugs and import-time failures** — the golden reference `base_*.py` file may itself contain bugs (e.g., using `tl.tanh` which doesn't exist in triton-ascend, API mismatches, or import-time dependencies on unavailable sibling/round files). Since `base_*.py` is read-only, wrap even `_load(BASE_FILE, ...)` in try/except, keep the required Baseline Triton2 column visible, and report `INFO ... unavailable_or_preskipped <ExceptionType>` / `inf` without interpolating raw exception text. Still produce `UNIT_TEST PASS` if the optimized kernel passes. Do not count a read-only base failure as an optimized failure.
- **Input baseline comparison failures in optimization tasks** — when the original `<N>_*.py` provider is used only as a comparison column and fails import/JIT due an already-identified unsupported API (for example `tl.tanh`) or would exceed the Ascend grid cap, do not print `FAIL` for that comparison provider; use `TEST Baseline Triton1 <label>: SKIP_UNAVAILABLE <ExceptionType> max_abs=inf` and an `inf` benchmark cell. `UNIT_TEST PASS` should be gated on the optimized provider and reference construction. If the task explicitly asks to repair the baseline provider, fix it instead of masking the failure.
- **Module-level model instantiation** — never instantiate `ModelNew(...)` at module level in `profile_kernels.py`. It creates on CPU and breaks when called with NPU input. Always use a lazy `_model()` cache that evaluates on first use and moves to NPU.
- **Benchmark no-grad for inference/cached paths** — if the optimized provider uses no-grad/inference caches (e.g. cached scaled BatchNorm affine vectors or folded Conv+BN weights), wrap the timed callable in `with torch.no_grad(): ...`. Otherwise autograd overhead and disabled cache branches can hide the intended production path.
- **2D kernel input construction** — some kernels (GELU, LogSoftmax, cumsum, etc.) operate on 2D tensors where `shape[-1]` is the reduction/feature dim. Read the kernel's `forward()` to understand input expectations before writing `_make_inputs()`. If the kernel does `M, N = x.shape` expecting truly 2D input, pass `(M, N)` tensors. If it flattens to 1D internally, 1D input is fine. Mismatched dimensionality causes wrong grid dims or incorrect results.
- **Style format** — plain named colours (`"blue"`, `"red"`, `"green"`, `"black"`), simple linestyles (`"-"`, `"--"`)
- **Shapes and tensors must match the operator requirement** — derive `_BENCH_SHAPES` and `_make_inputs()` from the kernel name/problem statement and original `get_inputs()` dimensions, dtype, layout, and data distribution. For large-K kernels keep K large; for small-K keep K small; for tall-skinny preserve the source skinny dimension; for irregular kernels include non-power-of-two/boundary shapes. `_BENCH_SHAPES` is the single source of truth for both unit tests and benchmark; run correctness on all benchmarked shapes, including the required benchmark shape, and cover both dispatch paths if the kernel has them.
- **Testing grid-cap/persistent paths without huge tensors** — when a persistent dispatch only triggers above 65,535 tiles, add a unit-only forced-persistent test instead of allocating enormous tensors. Pattern: import the optimized module, temporarily lower its `_MAX_GRID`/grid threshold (e.g. to `1`), clear the lazy model cache, run a modest shape that requires multiple tiles, compare to reference, then restore the threshold. This verifies persistent tile-loop correctness without poisoning the NPU context or running dead upstream work.
- **Testing Triton fallbacks hidden behind ACL production dispatch** — if the optimized host selects ACL/native ops in production but retains custom Triton direct/persistent fallback kernels, unit tests must temporarily disable the ACL flag (e.g. `_USE_ACL_DISPATCH = False`), clear the lazy model cache, and force both fallback paths. This keeps all dispatch paths tested while allowing the benchmark to report the real production path.
- **Bounded remote verification after timeouts** — if `remote_verify` times out because comparison providers or custom Triton fallback paths are too slow/toxic, make the next profile run *strictly bounded* before retrying: keep every provider column parser-visible, but pre-skip timeout-prone comparison providers on every shape with `TEST <Provider> <label>: SKIP_COMPARISON ... max_abs=inf` and `inf` benchmark cells; route the optimized provider through its safe production path (often ACL/native dispatch) for benchmark/correctness; keep separate forced-path unit tests for hidden Triton direct/persistent fallbacks on small tensors; and use manual low-rep `time.perf_counter + torch.npu.synchronize()` timing (e.g. warmup=1, rep=1 for target) instead of `do_bench` if needed. After a verify failure, `kernel_status` may move back to `optimize`; patch `profile_kernels.py`, call `kernel_status(action="advance")` to re-enter `verify`, then rerun `remote_verify`.
- **Device mismatch at import time** — module-level `ModelNew(...)` is CPU. Must be lazily moved to NPU
- **Duplicated torch_ref logic** — call `_run_torch_ref()`, don't rebuild inline in unit_test
- **baseline2 must map to `base_*.py`** — the golden reference, not a second copy of the input. If both `<N>_<KernelName>.py` and `base_<N>_<KernelName>.py` exist, `profile_kernels.py` MUST be a 4-line comparison: `PyTorch / ACL`, `Baseline Triton1` (`<N>_*.py`), `Baseline Triton2` (`base_<N>_<KernelName>.py`), `Optimized Triton`. The `base_*.py` file remains read-only: do not edit it, but importing/executing it as a comparison provider is required.
- **results.txt baseline2 parser visibility** — pipeline validation may reject a run when `base_*.py` exists but the captured output has no recognizable Baseline Triton2 TEST records, even if the benchmark table has the column. For every `_BENCH_SHAPES` label, print a TEST line for Baseline Triton2 using the canonical display name (or both display name and internal key), e.g. `TEST Baseline Triton2 <label>: SKIP/UNAVAILABLE ... max_abs=inf`; do not rely on `INFO test_unavailable baseline2 ...` or benchmark-only `inf` cells to satisfy validation.
- **Triton kernel name collision** — when loading multiple sibling files with same-named `@triton.autotune` kernels, use unique `k_<provider>_<stem>` module names, insert the module into `sys.modules` before `exec_module`, and load input before reference
- **Autotune decorator vs benchmark args** — keep `warmup`, `rep`, and `quantiles` in `triton.testing.do_bench` / `perf_report`, not in `@triton.autotune`. Passing benchmark-only kwargs to `@triton.autotune` can produce runtime errors such as `missing required positional argument: 'quantiles'`.
- **`inf`/`NaN` in speedup ratios** — filter with `np.isfinite() & (vals > 0)` before geomean. Use median-based ymax for bar charts. See `references/generate_report_robustness.md`
- **Weight-init matching** — torch_ref must replicate the EXACT same construction order with the SAME `torch.manual_seed(0)` as the baseline's `ModelNew.__init__`
- **Broadcasting mismatch** — for matmul kernels, use `torch.matmul()` directly instead of manual `unsqueeze`/`expand`
- **torch_npu.profiler sync requirement** — when using `torch_npu.profiler` as the benchmarking method (e.g. for FA or other ops where `do_bench` timing is unreliable), you MUST call `torch.npu.synchronize()` before entering the `with profile(...)` block, once before the for loop inside the block, and after EACH op call inside the for loop. Without per-iteration sync, device durations inflate ~10x (e.g. 159ms → 1401ms) because queued launches get attributed across step boundaries. See `references/torch-npu-profiler.md` for full API details.
- **Benchmark phase is gated on the unit test** — in the FA profiling template, if ANY provider/shape fails correctness, the run ends with `PROFILE_RESULT correctness_failed` and the benchmark phase NEVER runs (no timing lines at all). To obtain timings, comment the failing shapes out of `_BENCH_SHAPES` (or fix the root cause) and rerun; timings only flow after `UNIT_TEST PASS` / `PROFILE_RESULT ok`.

## Reference files

- `templates/profile_kernels.py` — Complete working template (3-line single-baseline format)
- `references/generate_report_parser_contract.md` — Strict format rules for `results.txt`
- `references/remote_execution_lessons.md` — Per-variant resilience, upload exclusion, entry-point requirement
- `references/profile-template-pitfalls.md` — Canonical single-process profiler shape, input-contract matching, correctness-gated benchmarking, `coreDim` poison handling, provider-wide `inf`, and Ascend `tl.dot` dtype pitfalls
- `references/torch-npu-profiler.md` — `torch_npu.profiler` API reference: `_ExperimentalConfig`, `ProfilerLevel`, `AiCMetrics`, `ExportType`, `schedule`, output file structure (`kernel_details.csv`, `operator_details.csv`, `op_statistic.csv`, `api_statistic.csv`, `step_trace_time.csv`), per-kernel pipeline breakdown, recommended configurations, and pitfalls. When adapting profile templates, prefer Level0 + `op_statistic.csv` Avg Time for low-overhead timing, call `torch.npu.synchronize()` before the profiler loop and after every profiled op before `prof.step()`, and keep canonical provider keys (`baseline1`, `baseline2`, `optimized`) in TEST lines for kernel-sandbox validation.
- `references/flash-attention-forward-profile.md` — FlashAttention forward profiling against `torch_npu.npu_fusion_attention`, bounded shape choices, tutorial-kernel Triton-Ascend compatibility fixes (`trans_b` → `tl.trans`, BLOCK=64, `num_stages=2`), and torch_npu dispatcher-op introspection.
## Measurement Methods (priority order)

1. **`torch_npu.profiler`** — preferred for device-level kernel durations. Use when you need
   per-kernel AICore pipeline breakdown (MAC ratio, MTE2 ratio, etc.) or when `do_bench` timing
   is unreliable. Requires sync between iterations. See `references/torch-npu-profiler.md`.
2. **`triton.testing.do_bench`** — preferred for quick wall-clock comparison. Returns ms directly.
3. **`time.perf_counter` + `torch.npu.synchronize()`** — fallback when neither of the above works.

## Constraints

- Generate this script AFTER optimized kernel is validated via cannsim
- This script runs on real NPU hardware, not cannsim
- Prefer `torch_npu.profiler` for device-level kernel durations; use `do_bench` for quick wall-clock comparison
- Return `triton.testing.do_bench(fn, warmup=25, rep=200, return_mode="mean")` directly (no scaling)
- **Concise output** — diagnosis and fix in 2-3 sentences
- **Do NOT modify `base_*.py` files** — they are the golden reference. Surface bugs via per-variant ERROR cells
