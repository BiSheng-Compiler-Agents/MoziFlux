import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Add lighter/wider mixes to improve occupancy on small N
        triton.Config({
            'BLOCK_SIZE_N': 64,
            'ROWS_PER_CTA': 8
        },
                      num_warps=1,
                      num_stages=2),
        triton.Config({
            'BLOCK_SIZE_N': 64,
            'ROWS_PER_CTA': 16
        },
                      num_warps=2,
                      num_stages=2),
        triton.Config({
            'BLOCK_SIZE_N': 64,
            'ROWS_PER_CTA': 32
        },
                      num_warps=2,
                      num_stages=3),
        triton.Config({
            'BLOCK_SIZE_N': 128,
            'ROWS_PER_CTA': 8
        },
                      num_warps=4,
                      num_stages=3),
        triton.Config({
            'BLOCK_SIZE_N': 128,
            'ROWS_PER_CTA': 16
        },
                      num_warps=4,
                      num_stages=4),
        triton.Config({
            'BLOCK_SIZE_N': 256,
            'ROWS_PER_CTA': 8
        },
                      num_warps=8,
                      num_stages=2),
        triton.Config({
            'BLOCK_SIZE_N': 256,
            'ROWS_PER_CTA': 16
        },
                      num_warps=8,
                      num_stages=3),
    ],
    key=['n_cols'],
)
@triton.jit
def _layernorm_gelu_scale_kernel(
    x_ptr,  # *[n_rows, n_cols]
    y_ptr,  # *[n_rows, n_cols] (stores to dtype(y_ptr))
    w_ptr,  # *[n_cols]
    b_ptr,  # *[n_cols]
    n_rows,  # total number of rows = prod(shape[:-1])
    n_cols,  # size of last dim
    inv_n_cols,  # 1.0 / n_cols
    eps,  # eps for layernorm
    scale,  # scaling factor after GELU
    ROWS_PER_CTA: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    cols = tl.arange(0, BLOCK_SIZE_N)
    col_mask = cols < n_cols

    # Load affine params once per-CTA into registers (fp32 for stability)
    gamma = tl.load(w_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
    beta = tl.load(b_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)

    inv_sqrt2 = 0.70710678118654752440084436210485  # 1/sqrt(2)

    # Persistent-CTA double-buffered prefetch over rows
    row0 = pid * ROWS_PER_CTA
    offs0 = row0 * n_cols + cols
    mask0 = (row0 < n_rows) & col_mask
    x_buf = tl.load(x_ptr + offs0, mask=mask0, other=0.0).to(tl.float32)

    for r in tl.static_range(ROWS_PER_CTA):
        row = row0 + r
        row_valid = row < n_rows

        # Use prefetched row
        x = x_buf

        # Prefetch next row to overlap memory latency with compute
        if r + 1 < ROWS_PER_CTA:
            next_row = row + 1
            offs_n = next_row * n_cols + cols
            mask_n = (next_row < n_rows) & col_mask
            x_buf = tl.load(x_ptr + offs_n, mask=mask_n,
                            other=0.0).to(tl.float32)

        # Compute statistics
        mu = tl.sum(x, axis=0) * inv_n_cols
        xc = x - mu
        var = tl.sum(xc * xc, axis=0) * inv_n_cols
        rstd = tl.math.rsqrt(var + eps)

        # Normalize + affine
        y = xc * rstd
        y = y * gamma + beta
        # Exact GELU
        y = 0.5 * y * (1.0 + tl.math.erf(y * inv_sqrt2))
        # Scale
        y = y * scale

        offs_store = row * n_cols + cols
        mask_store = row_valid & col_mask
        # Store in the dtype of y_ptr (implicit cast from fp32)
        tl.store(y_ptr + offs_store, y, mask=mask_store)
