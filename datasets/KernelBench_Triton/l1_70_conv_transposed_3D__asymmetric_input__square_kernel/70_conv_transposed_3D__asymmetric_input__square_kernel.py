import triton
import triton.language as tl

@triton.jit
def _flip_transpose_5d(
    inp_ptr,  # [Cin, Cout, Kd, Kh, Kw]
    out_ptr,  # [Cout, Cin, Kd, Kh, Kw]
    Cin: tl.constexpr,
    Cout: tl.constexpr,
    Kd: tl.constexpr,
    Kh: tl.constexpr,
    Kw: tl.constexpr,
    n_elements: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    base = pid * BLOCK
    offs = base + tl.arange(0, BLOCK)
    mask = offs < n_elements

    # Out tensor strides for [Cout, Cin, Kd, Kh, Kw]
    stride_out_kw = 1
    stride_out_kh = Kw
    stride_out_kd = Kh * Kw
    stride_out_ci = Kd * stride_out_kd
    stride_out_co = Cin * stride_out_ci

    co = offs // stride_out_co
    rem = offs - co * stride_out_co
    ci = rem // stride_out_ci
    rem = rem - ci * stride_out_ci
    kz = rem // stride_out_kd
    rem = rem - kz * stride_out_kd
    ky = rem // stride_out_kh
    kx = rem - ky * stride_out_kh

    in_kz = Kd - 1 - kz
    in_ky = Kh - 1 - ky
    in_kx = Kw - 1 - kx

    # In tensor strides for [Cin, Cout, Kd, Kh, Kw]
    stride_in_kw = 1
    stride_in_kh = Kw
    stride_in_kd = Kh * Kw
    stride_in_co = Kd * stride_in_kd
    stride_in_ci = Cout * stride_in_co

    in_idx = (
        ci * stride_in_ci
        + co * stride_in_co
        + in_kz * stride_in_kd
        + in_ky * stride_in_kh
        + in_kx * stride_in_kw
    )

    vals = tl.load(inp_ptr + in_idx, mask=mask, other=0)
    tl.store(out_ptr + offs, vals, mask=mask)
