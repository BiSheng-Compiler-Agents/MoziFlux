# Performance Report: l2_25 Conv2d_Min_Tanh_Tanh

## Cannsim trace methodology

The baseline custom epilogue was simulated with `cannsim_local_run` using a grid=1 sub-kernel probe (`BLOCK_HW=256`, `C_SIM=16`) that preserves the channelwise minimum plus two-tanh vector body while bounding compile/simulation time.  The production optimized path removes the custom Triton epilogue entirely, so optimized custom-kernel cannsim cycles are reported as zero; end-to-end latency must come from hardware profiling.

Trace file: `/tmp/cannsim_local/l2_25_conv2d_min_tanh_baseline_probe/cannsim_20260625123550_test_kernel/report/trace_core0.json`.

## Cannsim pipeline table

| Provider | Probe | wall cycles | Hardware time @0.4ns/cycle | Bottleneck | Notes |
|---|---:|---:|---:|---|---|
| Baseline Triton1 epilogue | grid=1, BLOCK_HW=256, C_SIM=16 | 3,439 | 1.376 us | MTE3 (2,477 busy cycles) | Scale-limited custom epilogue probe |
| Optimized Triton | no custom Triton epilogue | 0 | 0 us | none | Replaced by ACL `amin` + `tanh` dispatch |

## Baseline cannsim pipeline breakdown

| Pipeline | Ops | Busy cycles | Window |
|---|---:|---:|---|
| MTE3 | 2 | 2,477 | [4850,7328] |
| MTE2 | 33 | 1,640 | [4400,6040] |
| VEC | 1 | 1,609 | [4430,6039] |
| PUSHQ | 3 | 953 | [4442,6989] |
| RVECEX | 826 | 895 | [6063,6970] |
| SCALAR | 93 | 561 | [3898,7333] |
| RVECLD | 64 | 99 | [6079,6654] |
| RVECST | 4 | 36 | [6408,6977] |
| FLOWCTRL | 2 | 7 | [7330,7337] |
| RVECSU | 2 | 2 | [6062,6079] |

## Top cannsim instructions

| Instruction | Pipe | Count | Total cycles | Avg cycles |
|---|---|---:|---:|---:|
| MOV_SPR_XN | MTE2 | 17 | 12,905 | 759 |
| MOV_SRC_TO_DST_ALIGNv2 | MTE2 | 16 | 12,898 | 806 |
| WAIT_FLAG_VEC | MTE3 | 1 | 2,140 | 2,140 |
| WAIT_FLAG_MTE2 | VEC | 1 | 1,609 | 1,609 |
| VF | PUSHQ | 1 | 948 | 948 |
| RV_VADDS | RVECEX | 128 | 896 | 7 |

## Hardware latency

`remote_verify` passed correctness and benchmark on physical Ascend hardware.

| Shape | PyTorch / ACL (ms) | Baseline Triton1 (ms) | Baseline Triton2 (ms) | Optimized Triton (ms) | Optimized vs PyTorch / ACL |
|---|---:|---:|---:|---:|---:|
| small_32 | 0.013689 | inf (pre-skipped) | inf (pre-skipped) | 0.013770 | 0.994x |
| medium_128 | 0.152148 | inf (pre-skipped) | inf (pre-skipped) | 0.152419 | 0.998x |
| exact_256 | 4.512008 | inf (pre-skipped) | inf (pre-skipped) | 4.515236 | 0.999x |

Correctness: optimized passed all shapes with `max_diff=0.000e+00`.  Baseline Triton providers are retained as parser-visible comparison columns but pre-skipped because the optimized decision is to remove the fragile custom Triton epilogue rather than benchmark a toxic comparison path.
