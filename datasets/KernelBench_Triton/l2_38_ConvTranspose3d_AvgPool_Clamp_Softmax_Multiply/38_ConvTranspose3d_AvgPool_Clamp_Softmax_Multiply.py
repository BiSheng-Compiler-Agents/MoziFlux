import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_POS': 32}, num_warps=1, num_stages=2),
        triton.Config({'BLOCK_POS': 64}, num_warps=1, num_stages=2),
        triton.Config({'BLOCK_POS': 128}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_POS': 256}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_POS': 512}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_POS': 1024}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_POS': 2048}, num_warps=8, num_stages=5),
    ],
    key=['C', 'DHW'],
)
@triton.jit
def _clamp_softmax_mul2_tiled_ncdhw(
    x_ptr, y_ptr,
    N, C, DHW,
    stride_n, stride_c,
    clamp_min, clamp_max, scale,
    OUT_DTYPE: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_POS: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_tile = tl.program_id(1)

    # tile over positions within DHW
    pos_start = pid_tile * BLOCK_POS
    offs_p = pos_start + tl.arange(0, BLOCK_POS)
    mask_p = offs_p < DHW

    # channel lanes
    offs_c = tl.arange(0, BLOCK_C)
    mask_c = offs_c < C

    base_n = pid_n * stride_n

    # 2D tile pointer: [C, POS]
    ptrs = x_ptr + base_n + offs_c[:, None] * stride_c + offs_p[None, :]
    mask = mask_c[:, None] & mask_p[None, :]

    # Load and compute in fp32
    x = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)

    # clamp first
    x = tl.minimum(tl.maximum(x, clamp_min), clamp_max)

    # set masked lanes to -inf so they don't affect reductions across C
    neg_inf = -float('inf')
    x = tl.where(mask, x, neg_inf)

    # stable softmax over C (axis=0)
    x_max = tl.max(x, axis=0)
    x = tl.exp(x - x_max[None, :])
    denom = tl.sum(x, axis=0)
    inv = scale / denom[None, :]
    out = x * inv

    # write back
    tl.store(y_ptr + base_n + offs_c[:, None] * stride_c + offs_p[None, :], out.to(OUT_DTYPE), mask=mask)
