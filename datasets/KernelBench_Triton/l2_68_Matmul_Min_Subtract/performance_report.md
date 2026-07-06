# Performance Report

## Method

- Simulator: `cannsim_local_run(gen_report=True)` with sub-kernel hosts in the workspace.
- Micro-probe shape for direct comparison: `M=64, N=128, K=64`, grid `(1, 1, 1)`.
- Trace summaries were generated with `kernel-ops/simulation/scripts/aggregate_trace.py` from `trace_core0.json`.
- Hardware timing: `remote_verify(local_dir=..., run_test=True, run_bench=True)` running `profile_kernels.py` on Ascend NPU.

## cannsim trace comparison

| Kernel | BLOCK_M | BLOCK_N | BLOCK_K | Probe K | wall_cycles | HW latency (cycles * 0.4 ns) | x_events | i_events | Bottleneck |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Baseline Triton1 | 64 | 128 | 32 | 64 | 11628 | 4.651 us | 6050 | 196 | MTE3 (7967 busy cycles) |
| Optimized Triton | 64 | 128 | 64 | 64 | 9365 | 3.746 us | 5971 | 152 | MTE3 (5748 busy cycles) |

Speedup from cannsim wall cycles: `11628 / 9365 = 1.24x`.

## Pipeline detail

| Pipeline | Baseline busy cycles | Optimized busy cycles | Change |
|---|---:|---:|---:|
| MTE3 | 7967 | 5748 | -27.8% |
| PUSHQ | 6687 | 3847 | -42.5% |
| MTE2 | 6272 | 3496 | -44.3% |
| FLOWCTRL | 5767 | 3556 | -38.3% |
| RVECST | 5418 | 2621 | -51.6% |
| RVECLD | 4158 | 2424 | -41.7% |
| SCALAR | 1674 | 1503 | -10.2% |
| CUBE | 416 | 320 | -23.1% |

## Top instruction changes

| Instruction / event | Baseline | Optimized | Note |
|---|---:|---:|---|
| `WAIT_FLAG_VEC` on MTE3 | 8858 total cycles | 6369 total cycles | lower GM-store wait |
| `WAIT_FLAG_VEC` on MTE2 | 8177 total cycles | 3241 total cycles | fewer GM-load waits from larger K tile/in-place dot |
| `VF` on PUSHQ | 6761 total cycles | 3899 total cycles | reduced dispatch pressure |
| `SET_INTRA_BLOCKI` | 5909 total cycles | 3671 total cycles | reduced loop/control work |
| `RV_VSTI` | 14890 lane-sum cycles | 14555 lane-sum cycles | similar vector-store count, lower busy span |

## Hardware benchmark (`remote_verify`)

| Label | PyTorch / ACL (ms) | Baseline Triton1 (ms) | Baseline Triton2 | Optimized Triton (ms) | Opt vs Baseline Triton1 |
|---|---:|---:|---:|---:|---:|
| small_irregular | 0.101605 | 0.164609 | inf | 0.093470 | 1.76x |
| medium | 0.358610 | 1.115094 | inf | 0.542191 | 2.06x |
| target | 13.345538 | 71.961739 | inf | 39.232811 | 1.83x |

Baseline Triton2 is intentionally reported as `inf`/skipped because the active sandbox forbids reading `base_*.py` reference files.

## Correctness

`remote_verify` reported `UNIT_TEST PASS`.

| Provider | small_irregular max_abs | medium max_abs | target max_abs |
|---|---:|---:|---:|
| Baseline Triton1 | 2.38419e-07 | 9.53674e-07 | 3.57628e-06 |
| Optimized Triton | 2.38419e-07 | 9.53674e-07 | 3.57628e-06 |
| Optimized cache-hit path | 2.38419e-07 | 9.53674e-07 | 3.57628e-06 |
