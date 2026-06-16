import triton
import triton.language as tl


@triton.jit
def _reduce_bcdhw_to_b_kernel(
    x_ptr,  # *float32/float16/bfloat16, contiguous tensor [B, C]
    out_ptr,  # *float32/float16/bfloat16, tensor [B]
    stride_b,  # int, stride for batch dim of x in elements
    stride_c,  # int, stride for channel dim of x in elements
    C,  # int, channels
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < C
    vals = tl.load(x_ptr + pid * stride_b + offs * stride_c,
                   mask=mask,
                   other=0.0)
    total = tl.sum(vals.to(tl.float32), axis=0)
    tl.store(out_ptr + pid, total)
