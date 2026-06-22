# Performance Report

## Correctness and hardware verification

`remote_verify(run_test=True, run_bench=True)` passed on Ascend hardware.

```text
TEST optimized small_direct PASS
TEST optimized medium_direct PASS
TEST optimized target_direct PASS
UNIT_TEST PASS
```

Benchmark output (`profile_kernels.py`, milliseconds):

| label | PyTorch / ACL | Baseline Triton1 | Baseline Triton2 | Optimized Triton | Speedup vs PyTorch / ACL |
|---|---:|---:|---:|---:|---:|
| small_direct | 0.001599 | inf | inf | 0.001044 | 1.532x |
| medium_direct | 0.001953 | inf | inf | 0.001278 | 1.528x |
| target_direct | 0.023348 | inf | inf | 0.022468 | 1.039x |

Baseline comparison providers are kept parser-visible but skipped as `inf`: the editable input has a positional constructor incompatibility for `get_init_inputs()`, and the read-only reference is not modified.

## Cannsim sub-kernel traces

`cannsim_local_run` was run for one-core constant-fill probes with trace reports:

- Baseline bounded probe: `/tmp/cannsim_local/l2_13_fill_baseline2/.../trace_core0.json`, `BLOCK_SIZE=4096`.
- Optimized probe: `/tmp/cannsim_local/l2_13_fill_optimized2/.../trace_core0.json`, `BLOCK_SIZE=8192`.

| Path | Probe elements | Trace events | Wall cycles | Hardware latency (cycles × 0.4 ns) | Cycles / element | Bottleneck |
|---|---:|---:|---:|---:|---:|---|
| Baseline bounded fill | 4,096 | 211 | 5,613 | 2,245.2 ns | 1.370 | SCALAR / MTE3 |
| Optimized fill | 8,192 | 275 | 5,711 | 2,284.4 ns | 0.697 | SCALAR / MTE3 / RVECST |

Normalized per-element throughput improves `1.370 / 0.697 = 1.97x` by using the larger UB-safe tile.

## Cannsim pipeline breakdown

| Path | SCALAR | MTE3 | SCALARLDST | RVECST | PUSHQ | Other |
|---|---:|---:|---:|---:|---:|---:|
| Baseline bounded fill | 1,735 (35.3%) | 1,169 (23.8%) | 960 (19.5%) | 576 (11.7%) | 447 (9.1%) | 27 (0.5%) |
| Optimized fill | 1,732 (30.6%) | 1,279 (22.6%) | 958 (16.9%) | 1,152 (20.4%) | 512 (9.0%) | 27 (0.5%) |

Top instructions:

| Path | Top instruction costs |
|---|---|
| Baseline bounded fill | `LD_XD_XN_IMM=960`, `LDP_XI_XJ_XN=960`, `MOV_SRC_TO_DST_ALIGNv2=727`, `RV_VSTI=576`, `DC_PRELOAD_XN_IMM=494` |
| Optimized fill | `RV_VSTI=1152`, `LD_XD_XN_IMM=958`, `LDP_XI_XJ_XN=958`, `MOV_SRC_TO_DST_ALIGNv2=772`, `VF=508` |

Interpretation: the kernel is fixed-overhead and store-bound; increasing tile size improves work per launch/core without changing the dominant pipeline class.
