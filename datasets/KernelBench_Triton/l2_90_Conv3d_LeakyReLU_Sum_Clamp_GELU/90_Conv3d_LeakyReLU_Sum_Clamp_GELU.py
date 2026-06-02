import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 16384}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_SIZE": 8192}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=2),
    ],
    key=["n_elements"],
)
@triton.jit
def _fused_post_conv_kernel(
    x_ptr,          # *float32, input from conv: [N, C, D, H, W] flattened
    sum_ptr,        # *float32, per-channel bias: [C]
    y_ptr,          # *float32, output buffer (same shape as x)
    inner,          # int32, D*H*W
    C,              # int32, number of channels
    n_elements,     # int32, total number of elements in x
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # LeakyReLU with negative_slope=0.2
    neg = 0.2
    y = tl.where(x >= 0.0, x, x * neg)

    # Add per-channel bias (sum_tensor) with fast-path when block doesn't cross INNER boundary
    group0 = block_start // inner
    group1 = (block_start + (BLOCK_SIZE - 1)) // inner
    if group0 == group1:
        c_block = group0 % C
        sbias = tl.load(sum_ptr + c_block)  # scalar bias for the whole block
        y = y + sbias
    else:
        c_idx = (offsets // inner) % C
        vbias = tl.load(sum_ptr + c_idx, mask=mask, other=0.0)
        y = y + vbias

    # Clamp to [-1.0, 1.0]
    y = tl.maximum(tl.minimum(y, 1.0), -1.0)

    # GELU exact: 0.5 * x * (1 + erf(x / sqrt(2)))
    inv_sqrt2 = 0.7071067811865476
    t = y * inv_sqrt2
    erf_t = tl.erf(t)
    gelu = 0.5 * y * (1.0 + erf_t)

    tl.store(y_ptr + offsets, gelu, mask=mask)
