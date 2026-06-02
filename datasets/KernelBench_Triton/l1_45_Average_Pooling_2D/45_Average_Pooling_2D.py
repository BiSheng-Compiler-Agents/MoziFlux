import triton
import triton.language as tl

@triton.jit
def avg_pool2d_fwd_kernel(
    x_ptr, y_ptr,
    N, C, H, W, OH, OW,
    in_stride_n, in_stride_c, in_stride_h, in_stride_w,
    out_stride_n, out_stride_c, out_stride_h, out_stride_w,
    KH: tl.constexpr, KW: tl.constexpr,
    SH: tl.constexpr, SW: tl.constexpr,
    PH: tl.constexpr, PW: tl.constexpr,
):
    pid = tl.program_id(0)

    ow = pid % OW
    tmp = pid // OW
    oh = tmp % OH
    tmp = tmp // OH
    c = tmp % C
    n = tmp // C

    ih0 = oh * SH - PH
    iw0 = ow * SW - PW

    x_base = x_ptr + n * in_stride_n + c * in_stride_c
    y_offset = n * out_stride_n + c * out_stride_c + oh * out_stride_h + ow * out_stride_w

    acc = 0.0
    count = 0
    for kh in tl.static_range(0, KH):
        ih = ih0 + kh
        if (ih >= 0) and (ih < H):
            row_ptr = x_base + ih * in_stride_h
            for kw in tl.static_range(0, KW):
                iw = iw0 + kw
                if (iw >= 0) and (iw < W):
                    x_offset = row_ptr + iw * in_stride_w
                    acc += tl.load(x_offset).to(tl.float32)
                    count += 1

    out = acc / count
    tl.store(y_ptr + y_offset, out)
