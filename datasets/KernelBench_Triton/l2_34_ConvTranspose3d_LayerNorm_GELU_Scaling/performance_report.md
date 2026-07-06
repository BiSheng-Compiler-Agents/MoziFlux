# Performance Report

## Scope

`cannsim_local_run` was used on sub-kernel hosts for the fused LayerNorm + GELU + scale epilogue.  The simulator runs one representative core trace; it does not include the preceding `ConvTranspose3d` ACL kernel or full-shape host dispatch time.  Hardware latency below is computed from trace cycles using `0.4 ns/cycle`.

## Cannsim runs

| Variant | cannsim job | Kernel simulated | Tile rows | Trace |
|---|---:|---|---:|---|
| Baseline | `l2_34_baseline` | `_layernorm_gelu_scale_kernel` on contiguous NDHWC rows | 8 | `/tmp/cannsim_local/l2_34_baseline/cannsim_20260629174847_test_kernel/report/trace_core0.json` |
| Optimized | `l2_34_optimized` | `_layernorm_gelu_scale_ncdhw_kernel` on NCDHW rows | 16 | `/tmp/cannsim_local/l2_34_optimized/cannsim_20260629175103_test_kernel/report/trace_core0.json` |

Both runs built successfully and `cannsim report -n 0` produced `trace_core0.json`.

## Trace summary by pipeline

Durations are trace event `dur` sums.  The optimized tile has 2x rows, so per-row normalization is the most meaningful comparison.

| Pipeline | Baseline dur | Baseline % | Optimized dur | Optimized % | Notes |
|---|---:|---:|---:|---:|---|
| MTE2 | 8,611 | 21.9% | 42,512 | 41.2% | Optimized direct-NCDHW reads are strided across channel planes. |
| MTE3 | 3,698 | 9.4% | 16,184 | 15.7% | Optimized stores final NCDHW directly, avoiding a later full layout copy. |
| VEC | 969 | 2.5% | 12,904 | 12.5% | More rows per tile expose vector wait/compute events. |
| SCALARLDST | 3,648 | 9.3% | 9,904 | 9.6% | Row-to-NCDHW address decomposition adds scalar address work. |
| SCALAR | 9,647 | 24.5% | 9,163 | 8.9% | Persistent row-blocking amortizes scalar control over more rows. |
| RVECEX | 3,410 | 8.7% | 6,063 | 5.9% | Per-row vector execution is lower after row blocking. |
| PUSHQ | 7,852 | 19.9% | 3,238 | 3.1% | Fewer per-row launches/instruction queue pressure. |
| RVECLD | 1,087 | 2.8% | 2,300 | 2.2% | Local vector loads scale sublinearly with rows. |
| RVECST | 432 | 1.1% | 866 | 0.8% | Local vector stores scale roughly with rows. |
| FLOWCTRL | 9 | 0.0% | 9 | 0.0% | Same trace-level flow-control tail. |

## Latency comparison

| Metric | Baseline | Optimized | Change |
|---|---:|---:|---:|
| Trace wall cycles / tile | 12,257 | 16,459 | +34.3% |
| Rows per simulated tile | 8 | 16 | 2.00x |
| Trace wall cycles / row | 1,532.1 | 1,028.7 | **1.49x faster** |
| Hardware time / row (`cycles * 0.4 ns`) | 612.9 ns | 411.5 ns | **201.4 ns saved/row** |
| Total trace duration / row | 4,920.4 | 6,452.4 | NCDHW strided memory is more MTE-heavy in the sub-kernel. |

## Full-shape implication

For the default input, the output has `4,194,304` LayerNorm rows.  The input baseline's layernorm launch grid is at least `ceil(4,194,304 / 8) = 524,288`, exceeding Ascend's `65535` grid cap.  The optimized host interface uses the Triton direct-NCDHW kernel only below the cap and dispatches the default grid-cap regime to ACL-backed `layer_norm`/`gelu`, so the optimized path remains correct and benchmarkable while the input baseline is pre-skipped for that shape.

## Hardware profiling

`profile_kernels.py` contains the required `@triton.testing.perf_report` benchmark and unit test for:
- `PyTorch / ACL`
- `Baseline Triton1`
- `Baseline Triton2` when `base_*.py` exists (read-only comparison)
- `Optimized Triton`

Remote hardware verification was run successfully via `remote_verify` after profile generation.

| Label | PyTorch / ACL (ms) | Baseline Triton1 (ms) | Baseline Triton2 (ms) | Optimized Triton (ms) | Notes |
|---|---:|---:|---:|---:|---|
| `direct_small` | 0.410294 | 0.379279 | inf | **0.335984** | Optimized Triton direct NCDHW epilogue is 1.13x faster than Baseline Triton1 and 1.22x faster than PyTorch/ACL. |
| `gridcap_default` | 125.313812 | inf | inf | **124.906784** | Baseline Triton1 is pre-skipped because its grid exceeds `65535`; optimized dispatch uses ACL fallback and matches PyTorch/ACL latency. |

Correctness:
- `Optimized Triton/direct_small`: PASS, max_abs=0.000473499
- `Optimized Triton/gridcap_default`: PASS, max_abs=0
- Overall: `UNIT_TEST PASS`, `bench_passed=True`

Baseline2 exists but its read-only import depends on an unavailable `opt-round-4` file on the remote host, so `profile_kernels.py` keeps the required column and reports it as `inf`/unavailable without failing optimized verification.
