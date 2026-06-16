import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=2, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=4),
        triton.Config({}, num_warps=8, num_stages=2),
    ],
    key=["N"],
)
@triton.jit
def _fused_groupnorm_min_bias_kernel(
    x_ptr,  # [N, C]
    gamma_ptr,  # [C]
    beta_ptr,  # [C]
    bias_ptr,  # [C]
    out_ptr,  # [1, C, N, 1] - we index via strides over C and N
    N,
    C,  # sizes
    STRIDE_XN,  # stride between rows in x
    STRIDE_OC,  # stride over C in output
    STRIDE_ON,  # stride over N in output
    EPS,  # epsilon
    GROUP_SIZE: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    if pid >= N:
        return

    row_base = pid * STRIDE_XN

    # 2D channel indexing: [NUM_GROUPS, GROUP_SIZE] covering all channels
    offs_g = tl.arange(0, GROUP_SIZE)[None, :]  # [1, GS]
    offs_grp = tl.arange(0, NUM_GROUPS)[:, None]  # [NG, 1]
    ch_offs_2d = offs_grp * GROUP_SIZE + offs_g  # [NG, GS]

    # Load x and affine parameters (promote to f32 for numerics)
    x = tl.load(x_ptr + row_base + ch_offs_2d).to(tl.float32)  # [NG, GS]
    gamma = tl.load(gamma_ptr + ch_offs_2d).to(tl.float32)  # [NG, GS]
    beta = tl.load(beta_ptr + ch_offs_2d).to(tl.float32)  # [NG, GS]

    # Group-wise mean/var (unbiased=False)
    inv_gs = 1.0 / GROUP_SIZE
    sum1 = tl.sum(x, axis=1)  # [NG]
    sum2 = tl.sum(x * x, axis=1)  # [NG]
    mean = sum1 * inv_gs  # [NG]
    var = sum2 * inv_gs - mean * mean  # [NG]
    inv_std = tl.rsqrt(var + EPS)  # [NG]

    # Normalize + affine
    y = (x - mean[:, None]) * inv_std[:, None]  # [NG, GS]
    y = y * gamma + beta  # [NG, GS]

    # Row-wise min over all channels
    gmin = tl.min(y, axis=1)  # [NG]
    row_min = tl.min(gmin, axis=0)  # scalar

    # Add bias and write to output: O[0, c, n, 0] = bias[c] + row_min
    bias = tl.load(bias_ptr + ch_offs_2d).to(tl.float32)  # [NG, GS]
    out_tile = bias + row_min  # [NG, GS]
    out_ptrs = out_ptr + ch_offs_2d * STRIDE_OC + pid * STRIDE_ON
    tl.store(out_ptrs, out_tile)
