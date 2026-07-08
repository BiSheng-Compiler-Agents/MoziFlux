# Conv2d + GELU + GlobalAvgPool: ACL epilogue dispatch

## Pattern

Use this when an editable KernelBench-style baseline already runs `nn.Conv2d`/`F.conv2d` through ACL and then launches a custom Triton epilogue for exact GELU plus global spatial average pooling over `H,W`.

Prefer replacing only the epilogue with standard ACL/PyTorch operators:

```python
y = F.conv2d(x, weight, bias=bias, stride=stride,
             padding=padding, dilation=dilation, groups=groups)
return F.gelu(y, approximate="none").mean(dim=(-2, -1))
```

Keep constructor semantics, convolution parameters, input helpers, and dtype behavior aligned with the editable input kernel. Do not add new shape guards.

## Why

The custom Triton epilogue typically processes one `(N, C)` spatial plane per program, computes `tl.erf`-based GELU, then reduces `H*W` to one scalar. Even when correct, it adds a vector-core launch with MTE/SCALAR/RVEC work that duplicates mature ACL kernels. If cannsim shows MTE2/SCALAR/RVECEX or WAIT-heavy behavior for the epilogue microprobe, test ACL dispatch before deeper fusion.

## Cannsim/reporting guidance

- Simulate a sub-kernel for the removed epilogue only; shrink the spatial plane if needed (for example `TOTAL_HW=256`, `BLOCK_W=128`, `grid=(1,)`).
- Report optimized custom-kernel cycles as `0` only because the custom Triton launch is removed; do not claim cannsim measured ACL internals.
- Hardware latency must come from `remote_verify`/`profile_kernels.py`, compared against PyTorch/ACL and any safe editable baseline shapes.

## Profiling guidance

- Include small, medium, and exact/default spatial shapes.
- If the editable Triton baseline exact shape is too slow or risks timeout/context poisoning, keep its column visible and pre-skip only that timing cell with `inf` and a neutral `SKIP_COMPARISON`/`INFO` message.
- If sandbox/reference policy forbids reading `base_*.py`, keep `Baseline Triton2` parser-visible with skip lines rather than importing it.

## Evidence snapshot

A representative Conv2d -> GELU -> GlobalAvgPool case used `N=128,Cin=8,H=W=256,Cout=64,K=3`. The epilogue microprobe showed `wall_cycles=1838` with dominant MTE2/SCALAR/RVEC work. Replacing the custom epilogue with `F.gelu(...).mean((-2,-1))` removed the custom launch, passed optimized correctness with `max_abs=0`, and improved comparable hardware shapes by about 1.5-1.75x versus the editable Triton epilogue path while matching PyTorch/ACL at the exact/default shape within run noise.
