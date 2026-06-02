import triton
import triton.language as tl

@triton.jit
def _rowwise_lse_leaky_gelu2(
    x_ptr,           # pointer to [B, N] input
    y_ptr,           # pointer to [B, 1] output
    stride_xm,       # stride between rows for x
    stride_xn,       # stride between cols for x
    stride_ym,       # stride between rows for y
    NEG_SLOPE: tl.constexpr,  # leaky ReLU negative slope
    N: tl.constexpr,          # number of columns (out_features)
    BLOCK_N: tl.constexpr,    # tile size along N
):
    pid = tl.program_id(0)  # row id

    # Precompute arange once and base row pointer for better ILP
    r = tl.arange(0, BLOCK_N)
    row_ptr = x_ptr + pid * stride_xm

    # Streaming LogSumExp in a single pass for numeric stability and fewer global loads
    m = tl.full((1,), -float("inf"), dtype=tl.float32)  # running max
    s = tl.zeros((1,), dtype=tl.float32)                # running sum of exp shifted by m
    for n0 in range(0, N, BLOCK_N):
        offs = n0 + r
        mask = offs < N
        # Load masked lanes as -inf so we can drop further masking/where ops
        vals = tl.load(row_ptr + offs * stride_xn, mask=mask, other=-float("inf")).to(tl.float32)
        # tile max
        tile_max = tl.max(vals, axis=0)
        m_new = tl.maximum(m, tile_max)
        # accumulate sum in the new max domain
        s = s * tl.exp(m - m_new) + tl.sum(tl.exp(vals - m_new), axis=0)
        m = m_new

    # Final LogSumExp
    lse = m + tl.log(s)

    # Two LeakyReLU applications fused into one: for x<0 multiply by slope^2
    slope_sq = NEG_SLOPE * NEG_SLOPE
    x = tl.where(lse >= 0.0, lse, lse * slope_sq)

    # Two GELU (exact, erf-based) applications
    inv_sqrt2 = 0.7071067811865476  # 1/sqrt(2)
    x = 0.5 * x * (1.0 + tl.math.erf(x * inv_sqrt2))
    x = 0.5 * x * (1.0 + tl.math.erf(x * inv_sqrt2))

    # Store result
    y_offs = pid * stride_ym + tl.arange(0, 1)
    tl.store(y_ptr + y_offs, x)
