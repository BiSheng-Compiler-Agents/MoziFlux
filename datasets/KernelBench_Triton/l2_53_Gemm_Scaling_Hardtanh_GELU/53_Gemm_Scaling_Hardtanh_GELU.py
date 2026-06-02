import triton
import triton.language as tl

@triton.jit
def _scale_hardtanh_gelu_kernel(
    x_ptr,  # [rows, cols]
    y_ptr,  # [rows, cols]
    rows: tl.constexpr,
    cols: tl.constexpr,
    stride_x,  # stride between rows in elements
    stride_y,  # stride between rows in elements
    scale,
    minv,
    maxv,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)  # row id
    pid_n = tl.program_id(1)  # block along N

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m = pid_m

    # Boundary mask (keep even if grid matches to satisfy safety constraint)
    mask = (m < rows) & (offs_n < cols)

    # Pointers
    x_ptrs = x_ptr + m * stride_x + offs_n
    y_ptrs = y_ptr + m * stride_y + offs_n

    # Hint for the compiler on contiguity/align to enable vectorization
    tl.max_contiguous(offs_n, BLOCK_N)
    tl.multiple_of(offs_n, 16)

    # Fast path for full tiles to avoid masked memory ops on interior blocks
    n_start = pid_n * BLOCK_N
    full_tile = (n_start + BLOCK_N) <= cols

    # Stream from global to avoid polluting L1 for this pure epilogue
    x = tl.load(x_ptrs, cache_modifier=".cg") if full_tile else tl.load(x_ptrs, mask=mask, other=0.0, cache_modifier=".cg")

    # Compute in fp32 for numerical stability/accuracy parity with PyTorch GELU
    xf = x.to(tl.float32)
    # scale
    xf = xf * scale
    # hardtanh clamp
    xf = tl.minimum(tl.maximum(xf, minv), maxv)
    # exact GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
    inv_sqrt2 = 0.7071067811865476
    y32 = 0.5 * xf * (1.0 + tl.math.erf(xf * inv_sqrt2))

    # Cast back to original dtype for storage
    y = y32.to(x.dtype)

    # Store
    if full_tile:
        tl.store(y_ptrs, y)
    else:
        tl.store(y_ptrs, y, mask=mask)
