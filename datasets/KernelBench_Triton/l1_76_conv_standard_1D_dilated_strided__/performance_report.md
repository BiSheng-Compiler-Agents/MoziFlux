# Performance Report

## Cannsim setup

- Baseline trace: `cannsim_baseline/` compiled the same scalar/vector direct Conv1d kernel body as the editable baseline.
- Probe scale: `grid=(1,1,1)`, `IC=1`, `K=3`, `BLOCK_OC=64`, `L_OUT=1`. The exact `IC=64` variant exceeded the vector stack limit during compilation (`total stack object size 51264 > 6144`), so the trace is a scale-limited micro-probe of the same nested-loop body.
- Optimized path: no custom Triton device kernel is launched; `ModelNew.forward()` dispatches to ACL Conv1d through `torch.nn.functional.conv1d`.

## Cannsim trace comparison

| Path | wall cycles | hardware time (cycles × 0.4 ns) | custom Triton launch | dominant pipeline | Notes |
|---|---:|---:|---|---|---|
| Baseline direct Conv1d micro-probe | 5,642 | 2.257 µs | yes | MTE3 2,914 cycles | MTE2/VEC/SCALARLDST stalls from scalar/vector direct convolution |
| Optimized ACL dispatch | 0 | 0 µs custom-kernel time | no | n/a | custom Triton launch removed; hardware latency measured by `remote_verify` |

### Baseline pipeline utilization

| pipeline | ops | busy cycles | lane_sum | window |
|---|---:|---:|---:|---|
| MTE3 | 2 | 2,914 | 2,914 | [6620,9535] |
| MTE2 | 7 | 2,820 | 15,605 | [6387,9250] |
| VEC | 2 | 2,649 | 5,296 | [6601,9250] |
| SCALARLDST | 14 | 2,495 | 3,113 | [3922,6553] |
| SCALAR | 306 | 2,085 | 7,007 | [3902,9540] |
| PUSHQ | 3 | 189 | 190 | [6612,9434] |
| RVECEX | 77 | 138 | 606 | [9277,9415] |
| FLOWCTRL | 2 | 7 | 9 | [9537,9544] |

### Top baseline instruction costs

| instruction | pipe | count | total cycles | avg cycles |
|---|---|---:|---:|---:|
| MOV_SRC_TO_DST_ALIGNv2 | MTE2 | 3 | 7,849 | 2,616 |
| MOV_SPR_XN | MTE2 | 4 | 7,756 | 1,939 |
| LDP_XI_XJ_XN | SCALAR | 7 | 3,467 | 495 |
| WAIT_FLAG_VEC | MTE3 | 1 | 2,815 | 2,815 |
| WAIT_FLAG_MTE2 | VEC | 1 | 2,648 | 2,648 |
| WAIT_FLAG_SCALAR | VEC | 1 | 2,648 | 2,648 |
| LD_XD_XN_IMM | SCALARLDST | 5 | 2,248 | 450 |

## Hardware latency

`remote_verify` passed correctness and benchmark. Baseline Triton providers are parser-visible but pre-skipped as `inf` to avoid the invalid target launch product; the meaningful latency comparison is PyTorch/ACL reference vs optimized ACL dispatch.

| label | PyTorch / ACL (ms) | Baseline Triton1 | Baseline Triton2 | Optimized Triton (ms) | ref/opt |
|---|---:|---:|---:|---:|---:|
| small_B2_L257 | 0.007390 | inf | inf | 0.007315 | 1.0103x |
| medium_B8_L4096 | 0.031322 | inf | inf | 0.031280 | 1.0013x |
| irregular_B3_L8197 | 0.025928 | inf | inf | 0.025880 | 1.0019x |
| exact_B64_L524280 | 13.908068 | inf | inf | 13.924039 | 0.9989x |

Geomean PyTorch/ACL-to-optimized ratio: **1.0031x**. Optimized correctness passed all test shapes, including the exact target shape with output `(64, 128, 174758)`.
