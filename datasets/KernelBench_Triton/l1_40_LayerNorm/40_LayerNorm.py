import triton
import triton.language as tl

@triton.jit
def _layernorm_sums_kernel(
    x_ptr,         # (rows, M)
    sums_ptr,      # (rows,)
    sumsq_ptr,     # (rows,)
    M,             # int: number of features to normalize over
    BLOCK_SIZE: tl.constexpr,
):
    pid_row = tl.program_id(axis=0)
    pid_col = tl.program_id(axis=1)

    col_start = pid_col * BLOCK_SIZE
    offsets = col_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < M

    row_start = pid_row * M
    idx = row_start + offsets

    x = tl.load(x_ptr + idx, mask=mask, other=0.0).to(tl.float32)

    s = tl.sum(x, axis=0)
    s2 = tl.sum(x * x, axis=0)

    tl.atomic_add(sums_ptr + pid_row, s)
    tl.atomic_add(sumsq_ptr + pid_row, s2)

@triton.jit
def _layernorm_stats_kernel(
    sums_ptr,      # (rows,)
    sumsq_ptr,     # (rows,)
    mean_ptr,      # (rows,)
    rstd_ptr,      # (rows,)
    INV_M,         # float32 = 1.0 / M
    EPSILON,       # float32
):
    pid = tl.program_id(axis=0)
    s = tl.load(sums_ptr + pid).to(tl.float32)
    s2 = tl.load(sumsq_ptr + pid).to(tl.float32)
    mean = s * INV_M
    var = s2 * INV_M - mean * mean
    rstd = tl.rsqrt(var + EPSILON)
    tl.store(mean_ptr + pid, mean)
    tl.store(rstd_ptr + pid, rstd)

@triton.jit
def _layernorm_apply_kernel(
    x_ptr,         # (rows, M)
    w_ptr,         # (M,)
    b_ptr,         # (M,)
    mean_ptr,      # (rows,)
    rstd_ptr,      # (rows,)
    y_ptr,         # (rows, M)
    M,             # int
    BLOCK_SIZE: tl.constexpr,
):
    pid_row = tl.program_id(axis=0)
    pid_col = tl.program_id(axis=1)

    col_start = pid_col * BLOCK_SIZE
    offsets = col_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < M

    row_start = pid_row * M
    idx = row_start + offsets

    x = tl.load(x_ptr + idx, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + offsets, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    mean = tl.load(mean_ptr + pid_row)
    rstd = tl.load(rstd_ptr + pid_row)

    y = (x - mean) * rstd
    y = y * w + b
    tl.store(y_ptr + idx, y, mask=mask)
