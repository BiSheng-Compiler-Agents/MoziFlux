# Performance Report

## Cannsim setup

- Baseline trace: `/tmp/cannsim_local/kb42_maxpool2d_baseline_small/cannsim_20260624230615_test_kernel/report/trace_core0.json`
- Optimized trace: `/tmp/cannsim_local/kb42_maxpool2d_opt_flat_one/cannsim_20260624232511_test_kernel/report/trace_core0.json`
- Baseline sub-kernel: boundary tile, `BLOCK_HO=1`, `BLOCK_WO=16`.
- Optimized sub-kernel: one flat output element, `BLOCK=1`. A `BLOCK=16` optimized cannsim run compiled but did not become report-safe within the plugin's instr stability window, so the one-element trace is used for pipeline diagnosis.

## Cannsim wall latency

| Kernel | Wall cycles | Hardware time (ns, cycles × 0.4) | Trace events |
|---|---:|---:|---:|
| Baseline sub-kernel | 13,240 | 5,296.0 | 1,617 |
| Optimized flat sub-kernel | 7,363 | 2,945.2 | 580 |

## Pipeline busy-cycle breakdown

| Pipeline | Baseline busy cycles | Baseline % | Optimized busy cycles | Optimized % |
|---|---:|---:|---:|---:|
| SCALARLDST | 18,978 | 65.3% | 7,002 | 32.5% |
| SCALAR | 5,355 | 18.4% | 2,044 | 9.5% |
| PUSHQ | 2,212 | 7.6% | 390 | 1.8% |
| RVECEX | 1,242 | 4.3% | 1,230 | 5.7% |
| MTE3 | 806 | 2.8% | 875 | 4.1% |
| RVECLD | 320 | 1.1% | 159 | 0.7% |
| RVECST | 150 | 0.5% | 126 | 0.6% |
| MTE2 | 5 | 0.0% | 9,130 | 42.4% |

## Top instructions by total cycles

| Kernel | Instruction | Total cycles | Count |
|---|---|---:|---:|
| Baseline | `ST_XD_XN_IMM` | 11,326 | 190 |
| Baseline | `LD_XD_XN_IMM` | 5,854 | 266 |
| Baseline | `VF` | 2,184 | 5 |
| Baseline | `LD_XD_XN` | 1,798 | 64 |
| Optimized | `ST_XD_XN_IMM` | 6,456 | 28 |
| Optimized | `MOV_SRC_TO_DST_ALIGNv2` | 5,043 | 10 |
| Optimized | `MOV_SPR_XN` | 4,249 | 14 |
| Optimized | `WAIT_FLAG_VEC` | 713 | 3 |

## Hardware benchmark (`remote_verify`)

| label | PyTorch / ACL (ms) | Baseline Triton1 (ms) | Baseline Triton2 (ms) | Optimized Triton (ms) |
|---|---:|---:|---:|---:|
| direct_small | 0.003387 | inf | inf | 0.077180 |
| generic_2x2 | 0.003089 | inf | 0.005759 | 0.028418 |
| persistent_medium | 0.638336 | inf | inf | 81.752251 |
| target_original | 17.071651 | inf | inf | 2963.359619 |

Correctness: `UNIT_TEST PASS`; optimized passed `direct_small`, `generic_2x2`, `persistent_medium`, and `target_original`. Baseline1 fails Triton-Ascend compilation (`UnsupportedLanguageConstruct`); baseline2 is read-only and fails on most shapes with `MLIRCompilationError`.
