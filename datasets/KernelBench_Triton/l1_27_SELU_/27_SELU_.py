import triton
import triton.language as tl


@triton.jit
def _selu_kernel(
    x_ptr,
    y_ptr,
    n_elements,  # keep runtime arg to avoid recompiles on same shape
    ALPHA: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Coalescing & scheduling hints
    tl.max_contiguous(offsets, BLOCK_SIZE)
    tl.multiple_of(offsets, 16)

    mask = offsets < n_elements
    # Streaming load hint: we don't reuse x, prefer evict-first
    x = tl.load(x_ptr + offsets,
                mask=mask,
                other=0.0,
                eviction_policy='evict_first')
    x32 = x.to(tl.float32)

    # Precompute constants
    alpha_scale = SCALE * ALPHA

    # Branchless SELU:
    # pos = max(x, 0), neg = min(x, 0)
    # out = SCALE * pos + alpha_scale * (exp(neg) - 1)
    neg = tl.minimum(x32, 0.0)
    pos = tl.maximum(x32, 0.0)
    expm1_neg = tl.exp(neg) - 1.0
    out32 = pos * SCALE + expm1_neg * alpha_scale

    tl.store(y_ptr + offsets, out32.to(x.dtype), mask=mask)
