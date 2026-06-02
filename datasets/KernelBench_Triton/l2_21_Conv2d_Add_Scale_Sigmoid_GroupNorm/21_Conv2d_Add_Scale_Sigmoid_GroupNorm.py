import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 8192}, num_warps=8, num_stages=1),
    ],
    key=["n_elements"],
)
@triton.jit
def _bias_scale_sigmoid_kernel(
    x_ptr,            # *f32 [N, C, H, W] contiguous
    bias_ptr,         # *f32 [C, 1, 1] contiguous
    scale_ptr,        # *f32 [C, 1, 1] contiguous
    y_ptr,            # *f32 [N, C, H, W] contiguous
    HW: tl.constexpr, # H * W
    C: tl.constexpr,  # channels
    n_elements,       # total elements N*C*H*W
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # Load input
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)

    # Compute channel index for each element: c = (idx // HW) % C
    c_idx = (offs // HW) % C

    # Load bias and scale by channel
    b = tl.load(bias_ptr + c_idx, mask=mask, other=0.0)
    s = tl.load(scale_ptr + c_idx, mask=mask, other=0.0)

    # Fused: y = sigmoid((x + b) * s)
    z = (x + b) * s
    y = 1.0 / (1.0 + tl.exp(-z))

    # Store output
    tl.store(y_ptr + offs, y, mask=mask)
