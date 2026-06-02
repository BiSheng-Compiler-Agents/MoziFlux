import triton
import triton.language as tl

@triton.jit
def _instancenorm_divide_2d_fused_kernel(
    x_ptr,  # input/output
    N, C, H, W,
    stride_n, stride_c, stride_h, stride_w,
    eps, div_const,
    BLOCK_HW: tl.constexpr,
):
    pid = tl.program_id(axis=0)  # each program handles one (n, c)
    n = pid // C
    c = pid % C

    # Flattened spatial index [0, H*W)
    offs = tl.arange(0, BLOCK_HW)
    hw = H * W
    mask = offs < hw

    # Because the input is made contiguous by the caller, spatial slice is contiguous in memory.
    base = n * stride_n + c * stride_c
    ptrs = x_ptr + base + offs

    # Load values (masked), accumulate sum and sumsq in fp32
    x_vals = tl.load(ptrs, mask=mask, other=0.0)
    x_f32 = x_vals.to(tl.float32)

    sum_x = tl.sum(x_f32, axis=0)
    sum_x2 = tl.sum(x_f32 * x_f32, axis=0)

    # Compute mean and variance (population variance)
    inv_hw = tl.full((), 1.0 / hw, tl.float32)
    mean = sum_x * inv_hw
    var = sum_x2 * inv_hw - mean * mean
    var = tl.maximum(var, 0.0)

    inv_std = tl.rsqrt(var + eps)
    rcp_div = 1.0 / div_const
    scale = inv_std * rcp_div
    bias = -mean * scale

    # Normalize and divide-by in one pass; store back
    y = tl.fma(x_f32, scale, bias)
    tl.store(ptrs, y.to(x_vals.dtype), mask=mask)
