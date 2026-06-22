# Performance Report

## Cannsim setup

- Baseline sub-kernel: one `Cg=512` group tile from `_groupnorm_hardtanh_kernel`, `grid=(1,)`.
- Optimized sub-kernel: one `GROUP_BLOCK=4, Cg=512` tile from `_groupnorm_hardtanh_groupblock_kernel`, `grid=(1,)`.
- Hardware time conversion: `cycles * 0.4 ns`.

## Cannsim trace comparison

| Kernel | Work per program | Wall cycles | Sim time | Normalized cycles/group | Bottleneck |
|---|---:|---:|---:|---:|---|
| Baseline | 1 group | 6,013 | 2.405 us | 6,013 | PUSHQ 2,312 busy cycles |
| Optimized | 4 groups | 4,429 | 1.772 us | 1,107 | MTE3 2,347 busy cycles |

## Pipeline table

| Pipeline | Baseline busy | Optimized busy | Notes |
|---|---:|---:|---|
| PUSHQ | 2,312 | 1,105 | Fewer program-level vector dispatch events per group |
| SCALAR | 2,018 | 2,108 | Similar scalar setup amortized across 4 groups |
| SCALARLDST | 1,953 | 1,829 | Slightly lower despite 4x group work |
| MTE2 | 1,030 | 1,078 | Similar load cost, amortized per group |
| VEC | 906 | 920 | Similar wait cost, amortized per group |
| MTE3 | 819 | 2,347 | More stores in one program; per-group store cost still lower |
| RVECEX | 275 | 947 | More vector arithmetic because optimized program processes 4 groups |

## Hardware latency

`remote_verify` correctness: `UNIT_TEST PASS`; optimized direct and persistent dispatch paths passed.

| label | PyTorch / ACL (ms) | Baseline Triton1 (ms) | Baseline Triton2 (ms) | Optimized Triton (ms) | Speedup vs Baseline1 |
|---|---:|---:|---:|---:|---:|
| small | 0.012839 | 0.010173 | 0.008387 | 0.008422 | 1.208x |
| medium | 0.067867 | 0.096800 | 0.061993 | 0.067865 | 1.426x |
| irregular | 0.019721 | 0.022004 | 0.014860 | 0.016385 | 1.343x |
| target | 6.431207 | 6.787034 | 6.437325 | 6.499175 | 1.044x |

Geomean speedup vs editable baseline1: **1.247x**. Target optimized latency: **6.499175 ms**.
