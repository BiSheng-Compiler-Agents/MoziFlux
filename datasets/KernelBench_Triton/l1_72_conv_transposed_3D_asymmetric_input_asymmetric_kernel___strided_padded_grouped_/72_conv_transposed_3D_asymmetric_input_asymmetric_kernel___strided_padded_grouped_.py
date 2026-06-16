import triton
import triton.language as tl


@triton.jit
def _upsample3d_scatter_kernel(
    in_ptr,
    out_ptr,
    N,
    C,
    Di,
    Hi,
    Wi,
    Do,
    Ho,
    Wo,
    SD,
    SH,
    SW,
    in_stride_n,
    in_stride_c,
    in_stride_d,
    in_stride_h,
    in_stride_w,
    out_stride_n,
    out_stride_c,
    out_stride_d,
    out_stride_h,
    out_stride_w,
    BLOCK_W: tl.constexpr,
):
    # Simple compile-time sanity
    tl.static_assert(BLOCK_W > 0)
    # program ids
    pid_line = tl.program_id(axis=0)  # over lines (n,c,d,h)
    pid_wblk = tl.program_id(axis=1)  # block along width

    # Decompose pid_line into (n, c, d, h)
    CiHi = C * Di * Hi
    n = pid_line // CiHi
    rem = pid_line % CiHi
    c = rem // (Di * Hi)
    rem2 = rem % (Di * Hi)
    d = rem2 // Hi
    h = rem2 % Hi

    # Vector of width indices to process by this program instance
    w_off = pid_wblk * BLOCK_W + tl.arange(0, BLOCK_W)
    w_mask = w_off < Wi

    # Base pointers for input/output lines
    in_base = in_ptr + n * in_stride_n + c * in_stride_c + d * in_stride_d + h * in_stride_h
    out_d_idx = d * SD
    out_h_idx = h * SH
    out_base = out_ptr + n * out_stride_n + c * out_stride_c + out_d_idx * out_stride_d + out_h_idx * out_stride_h

    # Load input values and scatter them into strided output positions
    x = tl.load(in_base + w_off * in_stride_w, mask=w_mask, other=0.0)
    out_w_pos = w_off * SW
    tl.store(out_base + out_w_pos * out_stride_w, x, mask=w_mask)
