import triton
import triton.language as tl


@triton.jit
def _fused_post_ops_bias_kernel(
    x_ptr,  # *f32
    bias_ptr,  # *f32
    y_ptr,  # *f32
    n_elements,  # i32
    C,  # i32
    stride_c,  # i32 (elements)
    bias_stride_c,  # i32 (elements)
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offs = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # Hints to compiler for better vectorization/coalescing
    tl.multiple_of(block_start, BLOCK_SIZE)
    tl.max_contiguous(offs, BLOCK_SIZE)

    # Load input
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)

    # 1) ReLU
    x = tl.maximum(x, 0.0)

    # 2) LeakyReLU after ReLU is a no-op; omit to save work while preserving semantics

    # 3) GELU (exact): 0.5 * u * (1 + erf(u / sqrt(2)))
    inv_sqrt2 = 0.7071067811865476  # 1/sqrt(2)
    u = x
    e = tl.math.erf(u * inv_sqrt2)
    x = (u * (1.0 + e)) * 0.5

    # 4) Sigmoid: since x >= 0 after ReLU->GELU, use simplified stable form
    # Use exp2 for slightly faster evaluation: exp(-x) = exp2(-x * log2(e))
    LOG2E = 1.4426950408889634
    x = 1.0 / (1.0 + tl.exp2(-x * LOG2E))

    # Map each flattened position back to its channel for broadcast bias add.
    c_idx = ((offs // stride_c) % C).to(tl.int32)
    b = tl.load(bias_ptr + c_idx * bias_stride_c, mask=mask, other=0.0)
    out = x + b

    tl.store(y_ptr + offs, out, mask=mask)
