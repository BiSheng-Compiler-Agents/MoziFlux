import triton
import triton.language as tl

@triton.jit
def _fused_pointwise_ncdhw_kernel(
    x_ptr,           # *f32
    sf_ptr,          # *f32, shape [C]
    bias_ptr,        # *f32, shape [C]
    out_ptr,         # *f32
    n_elements,      # int
    C,               # int
    DHW,             # int = D*H*W
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offs = block_start + tl.arange(0, BLOCK_SIZE)
    m = offs < n_elements

    # Hints for better vectorization/coalescing
    tl.multiple_of(offs, 16)
    tl.max_contiguous(offs, 16)

    # Load input
    x = tl.load(x_ptr + offs, mask=m, other=0.0)

    c_idx = (offs // DHW) % C
    sf = tl.load(sf_ptr + c_idx, mask=m, other=0.0)
    b = tl.load(bias_ptr + c_idx, mask=m, other=0.0)

    # Fused pointwise:
    # 1) scale
    x = x * sf
    # 2) tanh(x) via 2*sigmoid(2x) - 1 (one exp)
    sig2x = 1.0 / (1.0 + tl.exp(-2.0 * x))
    x = 2.0 * sig2x - 1.0
    # 3) multiply by bias
    x = x * b
    # 4) sigmoid
    x = 1.0 / (1.0 + tl.exp(-x))

    # Store
    tl.store(out_ptr + offs, x, mask=m)
