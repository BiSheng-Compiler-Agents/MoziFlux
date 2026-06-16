import triton
import triton.language as tl


@triton.jit
def _sub_hswish_kernel(x_ptr, out_ptr, N, subtract_value,
                       BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    x = tl.load(x_ptr + offs, mask=mask, other=0).to(tl.float32)
    v = x - subtract_value
    # HardSwish: v * clamp(v + 3, 0, 6) / 6
    vp3 = v + 3.0
    vp3 = tl.minimum(tl.maximum(vp3, 0.0), 6.0)
    y = v * (vp3 * (1.0 / 6.0))
    tl.store(out_ptr + offs, y, mask=mask)


@triton.jit
def _mish_kernel(x_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    x = tl.load(x_ptr + offs, mask=mask, other=0).to(tl.float32)
    # Stable softplus: log1p(exp(-abs(x))) + max(x, 0)
    ax = tl.abs(x)
    sp = tl.log(1.0 + tl.exp(-ax)) + tl.maximum(x, 0.0)
    # tanh(sp) = (1 - exp(-2*sp)) / (1 + exp(-2*sp))
    e2 = tl.exp(-2.0 * sp)
    th = (1.0 - e2) / (1.0 + e2)
    y = x * th
    tl.store(out_ptr + offs, y, mask=mask)


@triton.jit
def _fused_hswish_maxpool_mish_kernel(
    x_ptr,  # *flat* input pointer (N*C*H*W)
    y_ptr,  # *flat* output pointer (N*C*H_out*W_out)
    N_OUT,  # total number of output elements
    N,
    C,
    H,
    W,  # input dims
    H_OUT,
    W_OUT,  # output spatial dims
    subtract_value,  # scalar
    K: tl.constexpr,  # pooling kernel size (square), stride = K
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N_OUT

    # Decompose flat output index -> (n, c, ho, wo)
    wo = offs % W_OUT
    t1 = offs // W_OUT
    ho = t1 % H_OUT
    t2 = t1 // H_OUT
    c = t2 % C
    n = t2 // C

    # Base input offset for the top-left of the pooling window
    in_h0 = ho * K
    in_w0 = wo * K
    base = ((n * C + c) * H + in_h0) * W + in_w0

    # Initialize running max with a very negative value for valid lanes
    maxv = tl.full([BLOCK_SIZE], -1e30, dtype=tl.float32)

    # Iterate over KxK window, apply HardSwish on-the-fly, then reduce max
    inv6 = 1.0 / 6.0
    for kh in tl.static_range(K):
        row_base = base + kh * W
        for kw in tl.static_range(K):
            v = tl.load(x_ptr + row_base + kw, mask=mask, other=0).to(
                tl.float32) - subtract_value
            vp3 = v + 3.0
            vp3 = tl.minimum(tl.maximum(vp3, 0.0), 6.0)
            hs = v * (vp3 * inv6)
            maxv = tl.maximum(maxv, hs)

    # Apply Mish on the pooled result for valid outputs only, avoid NaNs for invalid lanes
    mv = tl.where(mask, maxv, 0.0)
    ax = tl.abs(mv)
    sp = tl.log(1.0 + tl.exp(-ax)) + tl.maximum(mv, 0.0)
    e2 = tl.exp(-2.0 * sp)
    th = (1.0 - e2) / (1.0 + e2)
    outv = mv * th

    tl.store(y_ptr + offs, outv, mask=mask)
