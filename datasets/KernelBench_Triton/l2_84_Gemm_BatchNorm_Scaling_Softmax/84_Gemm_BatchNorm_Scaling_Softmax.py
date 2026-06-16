import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=2, num_stages=1),
        triton.Config({}, num_warps=4, num_stages=1),
        triton.Config({}, num_warps=8, num_stages=1),
        triton.Config({}, num_warps=16, num_stages=1),
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=2),
        triton.Config({}, num_warps=16, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=3),
    ],
    key=["N", "HAS_VECTOR_SCALE"],
)
@triton.jit
def _scale_softmax_row_kernel(
    x_ptr,  # *[B, N]
    s_ptr,  # *[1] or *[N]
    out_ptr,  # *[B, N]
    stride_xm,
    stride_xn,
    stride_om,
    stride_on,
    N,  # number of columns
    HAS_VECTOR_SCALE: tl.
    constexpr,  # 0 for scalar scale, 1 for per-column scale
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    # Row pointers
    x_row_ptr = x_ptr + pid * stride_xm + offs * stride_xn
    o_row_ptr = out_ptr + pid * stride_om + offs * stride_on

    # Hints for better codegen/coalescing
    tl.max_contiguous(offs, BLOCK_SIZE)
    tl.multiple_of(offs, 16)

    # Load logits to fp32
    x = tl.load(x_row_ptr, mask=mask, other=0.0,
                cache_modifier=".cg").to(tl.float32)

    LOG2E = 1.4426950408889634  # for exp2

    if HAS_VECTOR_SCALE:
        # Per-column scale; masked lanes neutral
        s = tl.load(s_ptr + offs, mask=mask, other=1.0,
                    cache_modifier=".cg").to(tl.float32)
        z = x * s
        # Ensure masked lanes don't affect reductions
        z = tl.where(mask, z, -float("inf"))
        m = tl.max(z, axis=0)
        e = tl.exp2((z - m) * LOG2E)
        den = tl.sum(e, axis=0)
        out = e * (1.0 / den)
    else:
        # Scalar scale; sign-aware centering to avoid extra reduction work
        s = tl.load(s_ptr).to(tl.float32)
        # Compute reductions ignoring masked lanes robustly
        pos_inf = float("inf")
        neg_inf = -float("inf")
        if s >= 0:
            x_for_max = tl.where(mask, x, neg_inf)
            mx = tl.max(x_for_max, axis=0)
            z = (x - mx) * s
        else:
            x_for_min = tl.where(mask, x, pos_inf)
            mn = tl.min(x_for_min, axis=0)
            z = (x - mn) * s
        # Mask OOB elements to -inf so they don't affect softmax
        z = tl.where(mask, z, neg_inf)
        e = tl.exp2(z * LOG2E)
        den = tl.sum(e, axis=0)
        out = e * (1.0 / den)

    tl.store(o_row_ptr, out, mask=mask)
