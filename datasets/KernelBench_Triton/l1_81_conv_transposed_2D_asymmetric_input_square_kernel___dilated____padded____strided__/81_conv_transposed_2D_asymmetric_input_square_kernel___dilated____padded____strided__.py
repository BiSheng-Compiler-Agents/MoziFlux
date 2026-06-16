import triton
import triton.language as tl


@triton.jit
def _flip_transpose_4d_kernel(
    inp_ptr,
    out_ptr,
    Cin: tl.constexpr,
    Cout: tl.constexpr,
    K: tl.constexpr,
    n_elements,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    base = pid * BLOCK
    offs = base + tl.arange(0, BLOCK)
    mask = offs < n_elements

    stride_out_kw = 1
    stride_out_kh = K
    stride_out_ci = K * K
    stride_out_co = Cin * stride_out_ci

    co = offs // stride_out_co
    rem = offs - co * stride_out_co
    ci = rem // stride_out_ci
    rem = rem - ci * stride_out_ci
    ky = rem // stride_out_kh
    kx = rem - ky * stride_out_kh

    in_ky = K - 1 - ky
    in_kx = K - 1 - kx

    stride_in_kw = 1
    stride_in_kh = K
    stride_in_co = K * K
    stride_in_ci = Cout * stride_in_co

    in_idx = (ci * stride_in_ci + co * stride_in_co + in_ky * stride_in_kh +
              in_kx * stride_in_kw)
    vals = tl.load(inp_ptr + in_idx, mask=mask, other=0.0)
    tl.store(out_ptr + offs, vals, mask=mask)


@triton.jit
def _stride_insert_zeros_2d_kernel(
    inp_ptr,
    out_ptr,
    N,
    C,
    H,
    W,
    stride_in_n,
    stride_in_c,
    stride_in_h,
    stride_in_w,
    stride_out_n,
    stride_out_c,
    stride_out_h,
    stride_out_w,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    n_elements,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    base = pid * BLOCK
    offs = base + tl.arange(0, BLOCK)
    mask = offs < n_elements

    stride_w_linear = 1
    stride_h_linear = W
    stride_c_linear = H * W
    stride_n_linear = C * stride_c_linear

    n = offs // stride_n_linear
    rem = offs - n * stride_n_linear
    c = rem // stride_c_linear
    rem = rem - c * stride_c_linear
    h = rem // stride_h_linear
    w = rem - h * stride_h_linear

    vals = tl.load(
        inp_ptr + n * stride_in_n + c * stride_in_c + h * stride_in_h +
        w * stride_in_w,
        mask=mask,
        other=0.0,
    )
    tl.store(
        out_ptr + n * stride_out_n + c * stride_out_c +
        (h * STRIDE_H) * stride_out_h + (w * STRIDE_W) * stride_out_w,
        vals,
        mask=mask,
    )
