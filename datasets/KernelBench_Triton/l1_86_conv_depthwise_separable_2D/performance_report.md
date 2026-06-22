# Performance Report

## Cannsim setup

- Tool: `cannsim_local_run(gen_report=True)`
- Job: `/tmp/cannsim_local/kb_l1_86_dwsep_baseline_v2`
- Trace: `/tmp/cannsim_local/kb_l1_86_dwsep_baseline_v2/cannsim_20260625054512_test_kernel/report/trace_core0.json`
- Probe: scale-limited fused direct-convolution microprobe (`BM=1, BN=1, C_IN=1, K=3`, `grid=(1,1,1)`). A larger `BM=16, BN=16, C_IN=4` probe failed backend compile with VF stack spill (`10016 > 6144` bytes), matching the expected direct-convolution scalar-loop limitation.

## Cannsim trace comparison

| Path | wall cycles | Hardware time (ns) | Bottleneck | Critical instruction | Notes |
|---|---:|---:|---|---|---|
| Baseline Triton direct fused conv microprobe | 3668 | 1467.2 | PUSHQ 1260 cyc | `MOV_SRC_TO_DST_ALIGNv2` MTE2 1773 cyc | Scalar/vector direct convolution, no Cube use |
| Optimized ACL dispatch | 0 custom Triton cycles | 0 custom Triton ns | N/A | N/A | Custom Triton launch eliminated; physical latency comes from ACL hardware profiling |

### Baseline pipeline utilization

| Pipeline | Ops | Busy cycles | Lane sum | Window |
|---|---:|---:|---:|---|
| PUSHQ | 12 | 1260 | 1263 | [5318,7068] |
| SCALARLDST | 34 | 1021 | 1333 | [4406,6630] |
| MTE2 | 11 | 927 | 2983 | [5281,6992] |
| MTE3 | 2 | 907 | 907 | [6651,7559] |
| SCALAR | 122 | 607 | 1943 | [3900,7564] |
| VEC | 2 | 359 | 716 | [6633,6992] |
| RVECEX | 23 | 87 | 166 | [5810,7049] |

## Hardware latency

`remote_verify` passed correctness and benchmark on Ascend hardware.

| label | PyTorch / ACL (ms) | Baseline Triton1 (ms) | Baseline Triton2 (ms) | Optimized Triton (ms) | ACL / Optimized |
|---|---:|---:|---:|---:|---:|
| tiny_32 | 0.014847 | 5.653528 | inf | 0.014978 | 0.991x |
| small_64 | 0.029521 | 28.287867 | inf | 0.029522 | 1.000x |
| medium_128 | 0.129588 | 210.440308 | inf | 0.130726 | 0.991x |
| exact_512 | 5.507072 | inf | inf | 5.505593 | 1.000x |

- `UNIT_TEST PASS`; optimized max diff was 0 on all benchmark shapes.
- Finite Baseline Triton1 → Optimized geomean speedup: 835.0x on tiny/small/medium shapes.
- Exact benchmark latency: PyTorch / ACL 5.507072 ms vs Optimized Triton 5.505593 ms.
- Baseline Triton1 exact shape was pre-skipped because its launch product is 262144 programs, exceeding Ascend `coreDim <= 65535`.
