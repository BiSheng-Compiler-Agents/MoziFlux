# Conv2d -> AvgPool2d -> Sigmoid -> Sum: ACL dispatch pattern

## Recognition

Use this when a KernelBench-style model runs `nn.Conv2d`/ACL first, then implements a standard post-conv chain in custom Triton:

```python
y = conv(x)
# custom Triton: avg pool over spatial windows -> sigmoid -> sum over C/H/W
```

Typical warning signs in the custom epilogue:
- NCHW index decomposition and pool-window address arithmetic dominate useful vector work.
- It writes a `(B, C)` partial tensor and launches a second channel/global reduction kernel.
- A cannsim grid=1 probe of the epilogue shows MTE/WAIT/SCALARLDST pressure rather than high RVEC throughput.

## Preferred optimization

Preserve convolution on ACL and route the standard post-conv operations through PyTorch/CANN:

```python
y = self.conv(x)
y = self.avg_pool(y)
y = torch.sigmoid(y)
return torch.sum(y, dim=(1, 2, 3))
```

This removes the custom pooling/activation/reduction Triton launches and lets ACL choose mature implementations for AvgPool2d, sigmoid, and reduction. It also preserves tuple/scalar `AvgPool2d` semantics instead of forcing a square-pool `K: tl.constexpr` path.

## Cannsim reporting

Trace the removed custom epilogue only; do not try to cannsim the full default convolution shape. A practical sub-kernel probe can shrink to one batch/channel plane and a small spatial tile (for example `B=1, C=1, H=8, W=64, K=4, BLOCK_W=16`) while preserving the NCHW/pool arithmetic bottleneck.

Report the optimized custom Triton cycles as `0` only if the custom Triton post-conv launch is completely removed. Hardware latency still comes from `remote_verify`, not cannsim.

## Profiling notes

- Include PyTorch/ACL, editable baseline, read-only baseline2, and optimized columns.
- If the active sandbox says `base_*.py` must not be read, keep Baseline Triton2 parser-visible with `SKIP_UNAVAILABLE ... max_abs=inf` records and do not import it.
- If the editable custom Triton baseline is too slow/toxic at the target shape, pre-skip only that comparison timing/correctness while still testing optimized correctness at the target shape.

## Evidence shape

A representative Conv2d + AvgPool2d + Sigmoid + Sum case had a baseline epilogue cannsim probe with `wall_cycles=11031`, bottleneck `MTE2` (`busy_cyc=7897`) plus large `WAIT_FLAG_MTE2` and `SCALARLDST` costs. Replacing the custom epilogue with ACL standard operators made optimized correctness exact and gave about 8-10x speedup versus the editable custom Triton baseline on small/irregular hardware shapes, with target latency close to the PyTorch/ACL reference.
