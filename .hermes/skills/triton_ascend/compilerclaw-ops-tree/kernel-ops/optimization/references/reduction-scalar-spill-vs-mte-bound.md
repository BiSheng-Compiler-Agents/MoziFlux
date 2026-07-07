# Reduction scalar-spill cleanup vs MTE-bound wall time

## Pattern

For row-wise norm/reduction kernels with loop-carried sums, use a one-lane tensor accumulator instead of a scalar accumulator when the reduction spans multiple `tl.range`/`while` iterations:

```python
# Better on Ascend: tensor state stays in vector registers
acc = tl.zeros((1,), dtype=tl.float32)
acc += tl.sum(vals, axis=0, keep_dims=True)

# Risky: scalar state can spill through SCALARLDST across loop iterations
acc = tl.zeros((), dtype=tl.float32)
acc += tl.sum(vals, axis=0)
```

This can sharply reduce `SCALARLDST` and `ST_XD_XN_IMM` trace cost.

## Interpretation caveat

A scalar-spill cleanup is not guaranteed to improve end-to-end latency if the kernel is dominated by GM traffic (`MTE2`/`MTE3`). Row-wise L1/L2/Frobenius normalization often needs two GM reads (first pass for denominator, second pass for scale/store), so trace improvements in scalar pipelines may show only small hardware gains.

## Reporting guidance

When this happens, document both facts:

- cannsim trace improvement, e.g. `SCALARLDST ops` and `ST_XD_XN_IMM total_cyc` decreased;
- wall/hardware latency remained MTE-bound, so speedup is modest.

Do not over-claim the optimization from instruction-level trace deltas alone.