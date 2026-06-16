import triton
import triton.language as tl


@triton.jit
def _fused_bias_residual_mul_add_3d(
    x_ptr,  # pointer to conv_transpose output, shape [N, C, D, H, W] flattened
    bias_ptr,  # pointer to bias per channel, shape [C]
    out_ptr,  # pointer to output tensor, same shape as x_ptr
    DHW,  # int: D*H*W
    C,  # int: number of channels
    BLOCK_K: tl.constexpr,
):
    # Program IDs
    pid_nc = tl.program_id(axis=0)  # over N*C groups
    pid_k = tl.program_id(axis=1)  # tiles along DHW

    # Offsets within DHW
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mask = offs_k < DHW

    # Base offset for this (n, c) group
    base = pid_nc * DHW

    # Load x and create original_x (detached in PyTorch, numerically same as x)
    x_val = tl.load(x_ptr + base + offs_k, mask=mask, other=0.0)
    original_x = x_val  # numerically identical to clone().detach()

    # Load per-channel bias
    c_idx = pid_nc % C
    b = tl.load(bias_ptr + c_idx)

    # Replicate the exact operation ordering:
    # x = x + bias
    x_val = x_val + b
    # x = x + original_x
    x_val = x_val + original_x
    # x = x * original_x
    x_val = x_val * original_x
    # x = x + original_x
    x_val = x_val + original_x

    # Store result
    tl.store(out_ptr + base + offs_k, x_val, mask=mask)
