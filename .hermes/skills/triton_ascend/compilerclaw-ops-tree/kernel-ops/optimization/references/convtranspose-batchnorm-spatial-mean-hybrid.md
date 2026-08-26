# ConvTranspose + BatchNorm + Spatial Mean-Subtract Hybrid Dispatch

Use this when a baseline already keeps `ConvTranspose*` and `BatchNorm*` on ACL/PyTorch, then applies a custom Triton epilogue that subtracts the mean over spatial dimensions per `(N,C)` plane.

## Recognition pattern

- Forward shape is typically `x = conv_transpose(x); x = batch_norm(x); y = x - mean_spatial(x)`.
- The custom Triton epilogue maps one program to a full `(N,C)` plane and scans `S = D*H*W` (or `H*W` / `L`) in one or more serial loops.
- Cannsim sub-kernel often shows VEC/MTE wait bottlenecks in the serial epilogue, suggesting parallel reduction.

## Preferred optimization

Preserve ACL for the convolution and batchnorm. Do **not** rewrite the convolution.

For the spatial mean-subtract epilogue, use hardware-guided hybrid dispatch:

```python
x = self.conv_transpose(x)
x = self.batch_norm(x)
x = x.contiguous()
S = x.shape[2] * x.shape[3] * x.shape[4]
planes = x.shape[0] * x.shape[1]

if S <= SMALL_S_THRESHOLD and planes <= 65535:
    y = torch.empty_like(x)
    _direct_mean_subtract_kernel[(planes,)](x, y, S, planes, BLOCK=2048)
    return y

return x - x.mean(dim=(2, 3, 4), keepdim=True)
```

The direct Triton path is only for tiny spatial planes where one launch wins. For medium/large planes, ACL/PyTorch mean reduction can beat a custom Triton partial-sum/finalize/subtract decomposition despite cannsim suggesting the serial baseline is inefficient.

## Pitfall: custom partial reduction may regress

A three-kernel custom epilogue (partial sums -> finalize -> subtract) adds launches and GM partial traffic. It can look plausible from a single partial-sum cannsim trace, but full hardware timing may be much slower than ACL mean on medium/target shapes. Treat this pattern as experimental unless hardware proves it wins.

## Profiling requirements

- Include at least one tiny shape that exercises the direct Triton path.
- Include medium and target shapes that exercise the ACL mean path.
- If testing a custom partial-reduction candidate, keep it as an A/B branch until `remote_verify` proves it faster; roll it back when it regresses.
- Report cannsim traces separately from hardware dispatch results: grid=1 traces diagnose the serial epilogue, but do not capture multi-launch overhead or ACL reduction speed.
