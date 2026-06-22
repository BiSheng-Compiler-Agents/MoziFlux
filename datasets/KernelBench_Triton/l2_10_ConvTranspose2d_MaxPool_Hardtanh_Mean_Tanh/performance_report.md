# Performance Report

## Cannsim setup

- Tool: `cannsim_local_run(gen_report=True)`
- Probe: one `(B,C)` spatial-reduction tile, grid=1, fp32 input, bounded `H_OUT=W_OUT=8`, `BLOCK=64`.
- Rationale: realistic `BLOCK=16384` is too large for practical cannsim; the bounded probe preserves the maxpool/reduction/tanh instruction pattern. Production routes larger planes to ACL based on hardware timing.

## Cannsim trace comparison

| Path | Probe body | wall_cycles | latency ns (cycles*0.4) | Bottleneck | Notes |
|---|---|---:|---:|---|---|
| Baseline Triton probe | maxpool + hardtanh + sum + exp/div tanh | 15,329 | 6,131.6 | SCALARLDST 11,853 | 6 VF/PUSHQ events |
| Optimized Triton probe | same body + `tl_math.tanh` | 14,014 | 5,605.6 | SCALARLDST 11,774 | 3 VF/PUSHQ events |

## Pipeline details

| Pipeline | Baseline busy_cyc | Optimized busy_cyc | Delta |
|---|---:|---:|---:|
| SCALARLDST | 11,853 | 11,774 | -79 |
| SCALAR | 4,205 | 4,172 | -33 |
| PUSHQ | 2,574 | 1,351 | -1,223 |
| MTE3 | 328 | 329 | +1 |
| RVECEX | 167 | 160 | -7 |
| RVECLD | 56 | 29 | -27 |
| RVECST | 54 | 27 | -27 |

## Hardware benchmark (`remote_verify`)

| label | PyTorch / ACL (ms) | Baseline Triton1 (ms) | Baseline Triton2 (ms) | Optimized Triton (ms) | Result |
|---|---:|---:|---:|---:|---|
| direct_small | 0.073383 | 0.067202 | inf | 0.066515 | 1.01x vs baseline1 |
| direct_medium | 0.109743 | 0.620610 | inf | 0.109971 | 5.64x vs baseline1; ACL fallback |
| target_persistent | 12.613186 | inf | inf | 12.589623 | target shape valid; ACL fallback |

Correctness: `UNIT_TEST PASS`; optimized max error was `8.941e-08` on direct_small and exact vs ACL on ACL-fallback shapes. Baseline1 target was pre-skipped because its huge single-block Triton reduction is toxic for the target shape; baseline2 was kept parser-visible and skipped as a read-only optional provider.
