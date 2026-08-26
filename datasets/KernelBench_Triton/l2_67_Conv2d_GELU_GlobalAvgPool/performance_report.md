# Performance Report

## Cannsim setup

- Tool: `cannsim_local_run(gen_report=True)`
- Job: `l2_67_gelu_gap_baseline`
- Trace: `/tmp/cannsim_local/l2_67_gelu_gap_baseline/cannsim_20260630022500_test_kernel/report/trace_core0.json`
- Microprobe: one GELU + global-average-pool spatial plane, `TOTAL_HW=256`, `BLOCK_W=128`, `grid=(1,)`.
- Optimized custom-kernel trace: not applicable; the optimized path removes the Triton epilogue and dispatches standard ACL operators.

## Cannsim trace comparison

| Variant | Custom Triton epilogue | Wall cycles | Simulated time | Dominant pipeline | Notes |
|---|---:|---:|---:|---|---|
| Baseline Triton1 epilogue | yes | 1,838 | 0.735 µs | MTE2 / SCALAR / RVECEX | Extra vector-core launch for GELU + spatial reduction |
| Optimized Triton | no | 0 | 0.000 µs | n/a | Custom launch removed; epilogue handled by ACL |

### Baseline pipeline breakdown

Percentages are busy cycles divided by trace wall cycles; overlap can make totals exceed 100%.

| Pipeline | Busy cycles | % of wall |
|---|---:|---:|
| MTE2 | 2,452 | 133.4% |
| SCALAR | 1,433 | 78.0% |
| RVECEX | 956 | 52.0% |
| VEC | 944 | 51.4% |
| PUSHQ | 189 | 10.3% |
| MTE3 | 152 | 8.3% |
| RVECLD | 48 | 2.6% |
| SCALARLDST | 27 | 1.5% |
| RVECST | 9 | 0.5% |
| FLOWCTRL | 9 | 0.5% |

### Top baseline instructions

| Cycles | Pipeline | Instruction |
|---:|---|---|
| 961 | MTE2 | `MOV_SRC_TO_DST_ALIGNv2` |
| 944 | VEC | `WAIT_FLAG_MTE2` |
| 741 | MTE2 | `MOV_SPR_XN` |
| 740 | MTE2 | `MOV_SRC_TO_DST_ALIGNv2` |
| 494 | SCALAR | `DC_PRELOAD_XN_IMM` |
| 480 | SCALAR | `LDP_XI_XJ_XN` |
| 177 | PUSHQ | `VF` |
| 151 | MTE3 | `MOV_SRC_TO_DST_ALIGNv2` |

## Hardware latency (`remote_verify`)

Correctness passed for the optimized path on all benchmark shapes with `max_abs=0`.

| label | PyTorch / ACL (ms) | Baseline Triton1 (ms) | Baseline Triton2 (ms) | Optimized Triton (ms) | Notes |
|---|---:|---:|---:|---:|---|
| small_32 | 0.173654 | 0.315187 | inf | 0.180027 | optimized 1.751× faster than Baseline Triton1 |
| medium_128 | 2.615391 | 4.027653 | inf | 2.615107 | optimized 1.540× faster than Baseline Triton1 |
| exact_256 | 85.133858 | inf | inf | 85.582291 | Baseline Triton1 pre-skipped to avoid timeout; optimized matches ACL within run noise |

## Correctness summary

| Provider | small_32 | medium_128 | exact_256 |
|---|---|---|---|
| Baseline Triton1 | PASS, max_abs=0.000123173 | PASS, max_abs=0.000123024 | SKIP_COMPARISON |
| Baseline Triton2 | SKIP_COMPARISON | SKIP_COMPARISON | SKIP_COMPARISON |
| Optimized Triton | PASS, max_abs=0 | PASS, max_abs=0 | PASS, max_abs=0 |
