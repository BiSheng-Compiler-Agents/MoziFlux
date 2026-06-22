# Optimizations

## 1. Middle-dimension 2D tile reduction

Baseline dim=1 maps one program to one `(b, n)` output and streams `M` with scalar-strided loads:

```python
pid = tl.program_id(axis=0)
b = pid // N
n = pid % N
ptrs = x_ptr + b * stride_b + n * stride_n + idx * stride_m
acc = tl.minimum(acc, tl.min(tl.load(ptrs, mask=mask, other=float("inf")), axis=0))
```

The optimized path maps one program to a contiguous N tile and reduces a `[BLOCK_M, BLOCK_N]` tile over M:

```python
vals = tl.load(x_ptr + base + m[:, None] * stride_m + n[None, :] * stride_n,
               mask=(m[:, None] < M) & (n[None, :] < N), other=float("inf"))
acc = tl.minimum(acc, tl.min(vals, axis=0))
tl.store(out_ptr + b * out_stride_b + n * out_stride_n, acc, mask=n < N)
```

Rationale: for the target `dim=1` reduction (`[128,4096,4095]`), this turns 4095 scalar-column kernels per batch into 32 contiguous N-tile kernels per batch, improving GM access coalescing and reducing dispatch count.

## 2. Grid-cap persistent dispatch

All optimized kernels launch at most Ascend's `65535` FFTS grid limit and loop over logical tiles/rows inside the kernel:

```python
n_programs = min(total_tiles, _MAX_GRID)
_kernel[(n_programs,)](..., total_tiles, n_programs, BLOCK_M=128, BLOCK_N=128)
# device: while tile < total_tiles: ...; tile += n_programs
```

Rationale: the original dim=1 target would launch `B*N = 524160` programs and overflow `coreDim`; the optimized target launches `128*ceil(4095/128)=4096` tile programs.

## 3. Dispatch coverage for dim=0/dim=2

Dim=0 uses the same N-tile reduction pattern over B; dim=2 uses a row-persistent last-dimension reduction. Both retain masks on every load/store and support non-power-of-two dimensions.

```python
# dim=0: reduce [BLOCK_B, BLOCK_N] -> [BLOCK_N]
# dim=2: persistent rows, reduce BLOCK_N chunks then tl.min(acc, axis=0)
```

Rationale: preserves the baseline `ModelNew(dim)` interface for all accepted dimensions while optimizing the target dim=1 path.
