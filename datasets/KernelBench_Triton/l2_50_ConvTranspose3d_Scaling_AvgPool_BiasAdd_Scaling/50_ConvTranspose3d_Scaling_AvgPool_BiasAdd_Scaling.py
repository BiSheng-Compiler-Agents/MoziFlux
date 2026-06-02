import triton
import triton.language as tl

@triton.jit
def _avgpool3d_k2s2_bias_scale_fused(
    x_ptr,                # *float32 [N, C, D, H, W] contiguous NCDHW
    bias_ptr,             # *float32 [C]
    y_ptr,                # *float32 [N, C, D2, H2, W2] contiguous NCDHW
    N, C, D, H, W,        # input dims
    D2, H2, W2,           # output dims = floor(D/2), floor(H/2), floor(W/2)
    scale1,               # float
    scale2,               # float
    n_elements,           # total output elements = N*C*D2*H2*W2
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offs = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # Fewer integer divisions for index de-linearization
    out_spatial = D2 * H2 * W2
    nc = offs // out_spatial                          # combined n*C + c
    rem = offs - nc * out_spatial
    t0 = rem // W2
    w2 = rem - t0 * W2
    d2 = t0 // H2
    h2 = t0 - d2 * H2
    c = nc % C                                        # for bias indexing

    # Map to input coordinates (stride=2, kernel=2, padding=0)
    w = w2 * 2
    h = h2 * 2
    d = d2 * 2

    # Strides for contiguous NCDHW
    stride_w = 1
    stride_h = W
    stride_d = H * W
    stride_nc = D * H * W

    # Linear base index for the (n, c, d, h, w) in input
    base = nc * stride_nc + d * stride_d + h * stride_h + w * stride_w

    # Offsets for the 2x2x2 kernel window
    o0 = 0
    o1 = 1
    o2 = stride_h
    o3 = stride_h + 1
    o4 = stride_d
    o5 = stride_d + 1
    o6 = stride_d + stride_h
    o7 = stride_d + stride_h + 1

    # Load 8 values
    x0 = tl.load(x_ptr + base + o0, mask=mask, other=0.0, cache_modifier=".ca")
    x1 = tl.load(x_ptr + base + o1, mask=mask, other=0.0, cache_modifier=".ca")
    x2 = tl.load(x_ptr + base + o2, mask=mask, other=0.0, cache_modifier=".ca")
    x3 = tl.load(x_ptr + base + o3, mask=mask, other=0.0, cache_modifier=".ca")
    x4 = tl.load(x_ptr + base + o4, mask=mask, other=0.0, cache_modifier=".ca")
    x5 = tl.load(x_ptr + base + o5, mask=mask, other=0.0, cache_modifier=".ca")
    x6 = tl.load(x_ptr + base + o6, mask=mask, other=0.0, cache_modifier=".ca")
    x7 = tl.load(x_ptr + base + o7, mask=mask, other=0.0, cache_modifier=".ca")

    # Pairwise reduction to shorten dependency chain
    s0 = x0 + x1
    s1 = x2 + x3
    s2 = x4 + x5
    s3 = x6 + x7
    s = (s0 + s1) + (s2 + s3)

    # Fuse scales to reduce arithmetic
    alpha = scale1 * scale2 * 0.125
    beta = scale2

    # Load channel bias and apply final affine transform
    b = tl.load(bias_ptr + c, mask=mask, other=0.0)
    out = s * alpha + b * beta

    # Store to output
    tl.store(y_ptr + offs, out, mask=mask)
