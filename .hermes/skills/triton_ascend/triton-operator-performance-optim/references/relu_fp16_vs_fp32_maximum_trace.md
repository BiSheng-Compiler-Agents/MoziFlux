# ReLU fp16 vs fp32 tl.maximum — cannsim Trace Evidence

Kernel: l1_19_ReLU, Ascend910_9589, BLOCK_SIZE=4096, grid=1 (sub-kernel host), fp16 input.
Date: June 2026.

## Baseline (fp16 tl.maximum + manual NaN handling)

```
wall_cycles: 3373
BOTTLENECK: 07_MTE3 (1592 busy_cyc)

Top instructions:
WAIT_FLAG_VEC   @ MTE3   dur=1226 cy  ← MTE3 store blocked waiting for VEC unit
WAIT_FLAG_MTE2  @ VEC    dur=1002 cy  ← VEC blocked waiting for MTE2 load
RV_VCMP_NE      @ RVECEX  cnt=64, total=384 cy   (x != x NaN check)
RV_VMAXS        @ RVECEX  cnt=64, total=384 cy   (tl.maximum fp16)
RV_VSEL         @ RVECEX  cnt=64, total=384 cy   (tl.where for NaN)
MOV_SRC_TO_DST_ALIGNv2 @ MTE2  dur=1001 cy
```

The 3-op NaN pattern (VCMP_NE + VMAXS + VSEL) each cost 384 cy.
WAIT_FLAG_VEC = 1226 cy: MTE3 cannot store until VEC fixed-function unit finishes.

## Optimized (fp32 upcast + propagate_nan=ALL)

```
wall_cycles: 3319  (-1.6% at sub-kernel scale)
BOTTLENECK: 02_SCALARLDST (1705 busy_cyc)  ← bottleneck shifted

WAIT_FLAG_VEC   @ MTE3   dur=1151 cy  (-6.1%)
WAIT_FLAG_MTE2  @ VEC    dur=986 cy   (-1.6%)
RV_VCVT_F2F     @ RVECEX  cnt=128, total=896 cy  (fp16→fp32 + fp32→fp16 casts)
RV_VCMP_NE, RV_VSEL absent  ← propagate_nan=ALL collapsed to 1 hardware instruction
```

WAIT_FLAG_VEC reduction: fp32 tl.maximum routes to RVECEX instead of VEC fixed-function unit.
The two cast ops (RV_VCVT_F2F ×128 = 7 cy/op) pipeline freely with MTE2/MTE3.

## Interpretation

Sub-kernel delta (-1.6%) understates real-hardware benefit because:
- At sub-kernel scale (1 tile), SCALARLDST (pointer setup) dominates regardless
- At production scale, the WAIT_FLAG_VEC serialization is a larger fraction of total time
- Episode 19 reported 7132 → 4838 cy (1.47×) on a real hardware Conv+ReLU+Bias kernel

## Key instruction mapping

| Triton construct | fp16 route | fp32 route |
|---|---|---|
| tl.maximum(x, 0.0) | VEC fixed unit (WAIT_FLAG_VEC stalls) | RVECEX (pipelines freely) |
| x != x (NaN check) | RVECEX: RV_VCMP_NE | eliminated (propagate_nan=ALL) |
| tl.where(nan, x, y) | RVECEX: RV_VSEL | eliminated (propagate_nan=ALL) |
| fp16→fp32 cast | — | RVECEX: RV_VCVT_F2F (7 cy/op, fast) |

## Conclusion

Always upcast to fp32 before tl.maximum on Ascend AIV. The two-cast roundtrip
(fp16→fp32→fp16) costs ~7 cy/op × 2 × BLOCK_SIZE but eliminates WAIT_FLAG_VEC
stalls that can be 1000+ cy. Net positive at any BLOCK_SIZE ≥ 128.
