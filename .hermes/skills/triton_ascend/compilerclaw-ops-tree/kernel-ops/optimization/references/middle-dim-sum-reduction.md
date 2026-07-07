# Middle-dimension sum reduction pattern

Use this when optimizing `torch.sum(x, dim=1, keepdim=True)`-style reductions over a 3D contiguous tensor `[B, M, N]`.

## Reusable pattern

If the baseline streams one M row at a time for each N tile, replace the per-row scalar/static loop with a 2D tile load and vector reduction across the M axis:

```python
# Slower row-streaming shape
for kk in tl.static_range(0, BLOCK_K):
    vals = tl.load(x + b*M*N + (m0 + kk) * N + n_offsets, mask=mask, other=0.0)
    acc += vals.to(tl.float32)

# Faster middle-dimension reduction shape
m = m0 + tl.arange(0, BLOCK_M)
n = n_tile * BLOCK_N + tl.arange(0, BLOCK_N)
vals = tl.load(
    x + b * M * N + m[:, None] * N + n[None, :],
    mask=(m[:, None] < M) & (n[None, :] < N),
    other=0.0,
).to(tl.float32)
acc += tl.sum(vals, axis=0)
```

When `M` is too large for one UB-resident tile, keep a single kernel and iterate `m0` in `tl.range(0, M, BLOCK_M)`; do not immediately introduce a multi-launch partial reduction unless hardware timing proves the extra GM partials and launch overhead are worth it.

## Dispatch

For large `B * ceil(N / BLOCK_N)`, prefer a 1D persistent/logical-tile loop capped by vector-core count and `65535`:

```python
n_programs = min(num_vectorcore, total_tiles, 65535)
_kernel[(n_programs,)](..., total_tiles, n_programs)

for tile in tl.range(pid, total_tiles, n_programs):
    b = tile // n_tiles
    n_tile = tile - b * n_tiles
    ...
```

Sub-kernel cannsim will show the per-tile instruction reduction but will not fully model the launch-count benefit; confirm final dispatch choice with hardware profiling.

## Example result from one session

For a 3D sum over `dim=1`, row streaming was MTE2-bound in cannsim. Changing to `[BLOCK_M, BLOCK_N]` tile reduction reduced sub-kernel wall cycles from 14,715 to 6,561 (2.24×) and hardware target latency from 21.039 ms to 8.159 ms (2.58× vs editable baseline).