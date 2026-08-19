# Post-linear row-normalization dispatch

Session pattern from a fused `F.linear` + row InstanceNorm/residual multiply kernel (`B=1024, K=N=8192`, fp32) on Ascend950.

## Durable lesson

When the main GEMM/linear path already dispatches to ACL and the custom Triton work is only a per-row normalization epilogue, first preserve the measured-fast arithmetic body. Small cannsim-only micro-optimizations to the row epilogue (for example rewriting arithmetic as `tl.fma` or toggling `care_padding=False`) can reduce some trace counters but still regress real hardware.

## Recommended pattern

- Keep the direct row-grid path for legal batch counts (`B <= 65535`).
- Add a separate persistent row-loop path only for `B > 65535` to avoid `coreDim` overflow.
- Treat the persistent path as a legality/generalization fix, not a target-shape performance win; grid=1 cannsim cannot show the dispatch-level benefit.
- If remote hardware shows a micro-optimization regresses, revert only that micro-change and keep the legality fallback if it is correct.

## Verification notes

Use cannsim for sub-kernel bottleneck classification, but decide target latency from `remote_verify`. Include a unit-test-only synthetic shape with large `B` and small feature dimensions to exercise the persistent path without distorting the benchmark table.
