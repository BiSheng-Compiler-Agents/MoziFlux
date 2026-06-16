import triton
import triton.language as tl


@triton.jit
def _sigmoid_scale_residual_kernel(x_ptr, out_ptr, n_elements, scale,
                                   BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Streaming load to reduce L1 pollution for this purely streaming op
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0, cache_modifier=".cg")
    # Compute in fp32 for stability/consistency with PyTorch
    x_fp32 = x.to(tl.float32)

    # Sigmoid and fused epilogue: y = x + scale * sigmoid(x)
    s = tl.sigmoid(x_fp32)
    y = x_fp32 + s * scale

    # Cast back to original dtype before store; streaming write
    y_cast = y.to(x.dtype)
    tl.store(out_ptr + offsets,
             y_cast,
             mask=mask,
             eviction_policy="evict_first")
