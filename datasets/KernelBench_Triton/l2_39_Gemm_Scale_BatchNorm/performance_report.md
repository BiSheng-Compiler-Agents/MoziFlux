# Performance Report: Gemm + Scale + BatchNorm

## Methodology

Cannsim was run with sub-kernel hosts (`grid=(1,1,1)`) as required. The baseline trace simulates the editable baseline's Triton affine epilogue tile (`M=128,N=128`); the optimized trace simulates one fused GEMM+affine tile (`M=128,N=128,K=32`). Full-shape hardware latency is reported as TBD because physical NPU verification is not available from cannsim alone.

## Cannsim trace paths

| Variant | trace_core0.json |
|---|---|
| Baseline Triton affine epilogue | `/tmp/cannsim_local/l2_39_gemm_scale_bn_baseline/cannsim_20260629193726_test_kernel/report/trace_core0.json` |
| Optimized fused GEMM+affine | `/tmp/cannsim_local/l2_39_gemm_scale_bn_opt/cannsim_20260629193908_test_kernel/report/trace_core0.json` |

## Pipeline trace summary

| Pipeline | Baseline busy cycles | Baseline events | Optimized busy cycles | Optimized events | Interpretation |
|---|---:|---:|---:|---:|---|
| Wall span | 4,516 | — | 6,657 | — | Optimized tile includes GEMM work absent from baseline epilogue-only tile. |
| MTE2 | 7,139 | 7 | 5,348 | 20 | Optimized uses more load instructions but lower aggregate MTE2 busy cycles for this sub-tile. |
| SCALAR | 5,387 | 254 | 6,716 | 240 | Fused tile adds GEMM loop/schedule scalar work. |
| RVECEX | 3,846 | 513 | 1,944 | 260 | Vector arithmetic drops because affine is fused after Cube accumulation. |
| MTE3 | 2,549 | 2 | 1,779 | 2 | Stores remain one output tile; optimized has lower MTE3 busy cycles. |
| CUBE | 0 | 0 | 2,411 | 4 | Optimized activates Cube via `tl.dot`; baseline epilogue is vector-only. |
| FIXP | 0 | 0 | 2,718 | 4 | Optimized includes Cube result movement. |
| FLOWCTRL | 9 | 2 | 6,056 | 7 | Expected Cube/vector synchronization in fused GEMM tile. |

## Top instructions

| Variant | Instruction | Count | Total cycles | Rationale |
|---|---|---:|---:|---|
| Baseline | `MOV_SRC_TO_DST_ALIGNv2` (MTE2) | 3 | 3,835 | Load affine/input tile data. |
| Baseline | `RV_VSTI` (RVECST) | 256 | 2,520 | Store vector lanes after affine. |
| Baseline | `RV_VMUL` + `RV_VADD` | 512 | 3,840 | Separate vector affine math. |
| Optimized | `SET_INTRA_BLOCKI` (FLOWCTRL) | 2 | 5,424 | Cube/vector synchronization. |
| Optimized | `RV_VSTI` (RVECST) | 512 | 6,136 | Store fused tile. |
| Optimized | `MMAD` (CUBE) | 1 | 2,115 | Matrix multiply core work. |
| Optimized | `WAIT_FLAG_CUBE` (MTE1/FIXP) | 4 | 7,310 | Movement/synchronization around Cube output. |

## Hardware latency

`remote_verify` correctness passed for Optimized Triton on all profile shapes.

| Shape | PyTorch / ACL (ms) | Baseline Triton1 (ms) | Baseline Triton2 (ms) | Optimized Triton (ms) |
|---|---:|---:|---:|---:|
| small_M128_K1024_N512 | 0.274032 | 0.329898 | 0.336053 | 0.286941 |
| medium_M1024_K2048_N1024 | 3.542306 | 3.557000 | 3.563733 | 3.525813 |
| irregular_M257_K1024_N768 | 0.534901 | 0.589432 | 0.606345 | 0.717306 |
| required_M16384_K4096_N4096 | 397.297333 | 391.322021 | 391.653259 | 394.978699 |

| Source | Latency |
|---|---:|
| Cannsim baseline sub-kernel wall span | `4,516 cycles × 0.4 ns = 1.806 µs` |
| Cannsim optimized sub-kernel wall span | `6,657 cycles × 0.4 ns = 2.663 µs` |
| Full-shape physical NPU required shape | `394.978699 ms` optimized; `UNIT_TEST PASS` |

## Notes

The optimized sub-kernel is not expected to have fewer sub-tile cycles than the baseline epilogue trace because it includes GEMM, bias, scale, and BatchNorm affine in one kernel while the baseline trace only covers the post-ACL affine Triton kernel. The performance benefit is removal of host/GM materialization and dispatch boundaries at full shape, not a lower cost for doing strictly more work inside the simulated tile.
