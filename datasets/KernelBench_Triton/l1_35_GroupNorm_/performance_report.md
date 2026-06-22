# Performance Report — 35_GroupNorm_

## Hardware benchmark (`remote_verify`, Ascend NPU)

`UNIT_TEST PASS` for baseline1, baseline2, optimized, and the optimized persistent-dispatch path.

| label | PyTorch / ACL (ms) | Baseline Triton1 (ms) | Baseline Triton2 (ms) | Optimized Triton (ms) | Speedup vs Baseline1 |
|---|---:|---:|---:|---:|---:|
| small_direct | 0.004783 | 0.034247 | 0.003587 | 0.036312 | 0.94x |
| medium_direct | 0.023302 | 0.947628 | 0.025211 | 0.569545 | 1.66x |
| benchmark_persistent | 15.645546 | 520.985474 | 16.206888 | 479.059418 | 1.09x |

## Cannsim trace comparison

All cannsim runs used sub-kernel hosts and generated `trace_core0.json` with `cannsim_local_run(gen_report=True)`. Hardware time estimate uses `cycles * 0.4 ns`.

### Stats phase, one 16-element reduction tile

| kernel | trace path | wall cycles | est. hw ns | bottleneck | top critical instruction |
|---|---|---:|---:|---|---|
| baseline stats | `/tmp/cannsim_local/kb35_baseline_stats_b16/.../trace_core0.json` | 5823 | 2329.2 | PUSHQ 3369 cyc | `VF` 3285 cyc |
| optimized partial stats | `/tmp/cannsim_local/kb35_opt_partial_small2/.../trace_core0.json` | 3676 | 1470.4 | PUSHQ 1836 cyc | `VF` 1794 cyc |

Stats sub-kernel delta: `5823 -> 3676` cycles, 1.58x per partial tile. Full-shape benefit comes from exposing up to 32 reducers per `(N, group)` instead of serializing an entire group in one program.

### Apply phase, 1024 spatial elements

| kernel | trace path | wall cycles | est. hw ns | bottleneck | top critical instruction |
|---|---|---:|---:|---|---|
| baseline apply (`BLOCK_HW=256`, four loop chunks) | `/tmp/cannsim_local/kb35_baseline_fwd_hw1024/.../trace_core0.json` | 5368 | 2147.2 | MTE3 3153 cyc | `WAIT_FLAG_VEC` 7167 cyc total |
| optimized apply (`BLOCK_HW=1024`, one loop chunk) | `/tmp/cannsim_local/kb35_opt_apply_channel1024/.../trace_core0.json` | 3439 | 1375.6 | SCALARLDST 2174 cyc | `LD_XD_XN` 3049 cyc total |

Apply sub-kernel delta: `5368 -> 3439` cycles, 1.56x. The improvement comes from removing three loop chunks for each 1024-element spatial span.

## Verification artifacts

- `results.txt` contains the exact remote correctness and benchmark output.
- Cannsim reports generated successfully for baseline stats, optimized partial stats, baseline apply, and optimized apply.
