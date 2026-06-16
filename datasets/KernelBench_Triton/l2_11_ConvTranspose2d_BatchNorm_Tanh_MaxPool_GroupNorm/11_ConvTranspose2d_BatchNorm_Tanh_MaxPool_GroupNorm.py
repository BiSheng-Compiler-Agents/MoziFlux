import triton
import triton.language as tl


@triton.jit
def _tanh_maxpool2x2_nchw_kernel(
    x_ptr,
    y_ptr,
    N: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    H_OUT: tl.constexpr,
    W_OUT: tl.constexpr,
    BLOCK_HO: tl.constexpr,
    BLOCK_WO: tl.constexpr,
):
    pid_nc = tl.program_id(0)
    pid_ho = tl.program_id(1)
    pid_wo = tl.program_id(2)

    n = pid_nc // C
    c = pid_nc % C

    ho_offsets = pid_ho * BLOCK_HO + tl.arange(0, BLOCK_HO)
    wo_offsets = pid_wo * BLOCK_WO + tl.arange(0, BLOCK_WO)

    ho_mask = ho_offsets < H_OUT
    wo_mask = wo_offsets < W_OUT
    HO = ho_offsets[:, None]
    WO = wo_offsets[None, :]
    out_mask = ho_mask[:, None] & wo_mask[None, :]

    base_x = (n * C + c) * (H * W)
    base_y = (n * C + c) * (H_OUT * W_OUT)
    y_offs = base_y + HO * W_OUT + WO

    ih0 = HO * 2
    iw0 = WO * 2
    row0 = base_x + ih0 * W
    row1 = row0 + W

    offs00 = row0 + iw0
    offs01 = offs00 + 1
    offs10 = row1 + iw0
    offs11 = offs10 + 1

    v00 = tl.load(x_ptr + offs00, mask=out_mask, other=-float("inf"))
    v01 = tl.load(x_ptr + offs01, mask=out_mask, other=-float("inf"))
    v10 = tl.load(x_ptr + offs10, mask=out_mask, other=-float("inf"))
    v11 = tl.load(x_ptr + offs11, mask=out_mask, other=-float("inf"))

    # Pool over the already-activated tensor.
    m0 = tl.maximum(v00, v01)
    m1 = tl.maximum(v10, v11)
    mp = tl.maximum(m0, m1)
    tl.store(y_ptr + y_offs, mp, mask=out_mask)
