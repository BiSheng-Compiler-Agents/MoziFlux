import triton
import triton.language as tl

@triton.jit
def _sumsq_kernel(x_ptr, n_elements, out_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK
    offsets = block_start + tl.arange(0, BLOCK)
    tl.max_contiguous(offsets, BLOCK)
    mask = offsets < n_elements
    # Load as source dtype then accumulate in fp32
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    s = tl.sum(x * x, axis=0)
    # Accumulate per-CTA partial sum into a single global scalar
    tl.atomic_add(out_ptr, s)

@triton.jit
def _reduce_partials_kernel(partials_ptr, n_partials, out_ptr, BLOCK: tl.constexpr):
    # Kept for compatibility (unused in this optimized path)
    offs = tl.arange(0, BLOCK)
    acc = 0.0
    idx = 0
    while idx < n_partials:
        offsets = idx + offs
        mask = offsets < n_partials
        vals = tl.load(partials_ptr + offsets, mask=mask, other=0.0)
        acc += tl.sum(vals, axis=0)
        idx += BLOCK
    tl.store(out_ptr, acc)

@triton.jit
def _scale_kernel(x_ptr, y_ptr, n_elements, sumsq_ptr, BLOCK: tl.constexpr):
    # Load global sum of squares and compute inverse Frobenius norm
    ss = tl.load(sumsq_ptr)  # float32
    inv_norm = tl.rsqrt(ss)
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK
    offsets = block_start + tl.arange(0, BLOCK)
    tl.max_contiguous(offsets, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = x.to(tl.float32) * inv_norm
    tl.store(y_ptr + offsets, y, mask=mask)
