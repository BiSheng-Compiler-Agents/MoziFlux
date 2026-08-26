# Middle-dimension min reduction (`[B,M,N]`, reduce dim=1)

Use this when a 3D contiguous tensor reduces over `M` and the baseline maps one program to one `(b,n)` output.

## Pattern

Replace scalar-column streaming:

```python
# one output column per program
ptrs = x_ptr + b * stride_b + n * stride_n + idx * stride_m
acc = tl.minimum(acc, tl.min(tl.load(ptrs, mask=mask, other=float("inf")), axis=0))
```

with an N-tile reduction over a `[BLOCK_M, BLOCK_N]` tile:

```python
offs_m = tl.arange(0, BLOCK_M)
offs_n = tl.arange(0, BLOCK_N)
while tile < total_tiles:
    b = tile // n_tiles_n
    n = (tile - b * n_tiles_n) * BLOCK_N + offs_n
    n_mask = n < N
    acc = tl.full((BLOCK_N,), float("inf"), dtype=tl.float32)
    m0 = 0
    while m0 < M:
        m = m0 + offs_m
        mask = (m[:, None] < M) & n_mask[None, :]
        vals = tl.load(x_ptr + b * stride_b + m[:, None] * stride_m + n[None, :] * stride_n,
                       mask=mask, other=float("inf"))
        acc = tl.minimum(acc, tl.min(vals, axis=0))
        m0 += BLOCK_M
    tl.store(out_ptr + b * out_stride_b + n * out_stride_n, acc, mask=n_mask)
    tile += n_programs
```

## Dispatch

Flatten `(B, ceil(N/BLOCK_N))` to one 1D logical tile space and cap the physical launch:

```python
n_tiles_n = triton.cdiv(N, BLOCK_N)
total_tiles = B * n_tiles_n
n_programs = min(total_tiles, 65535)
_kernel[(n_programs,)](..., total_tiles, n_tiles_n, n_programs,
                       BLOCK_M=128, BLOCK_N=128)
```

This avoids `coreDim > 65535` for target shapes where `B*N` scalar-output programs overflow.

## Reporting cannsim fairly

A scalar baseline sub-kernel may compute 1 output while the optimized tile computes `BLOCK_N` outputs. Normalize baseline cycles and pipeline counts by `BLOCK_N` when comparing same logical work, and state the normalization explicitly in `performance_report.md`.

## Caveats

- `+inf` is the neutral padding value for min reduction.
- Tune `BLOCK_M/BLOCK_N` on hardware; larger tiles reduce launch/loop overhead but can increase vector and MTE3 wait costs.
- Keep dim=0/dim=2 fallback or specialized paths if the host interface accepts arbitrary `dim`; do not optimize only the benchmark dim unless the original contract restricts it.
