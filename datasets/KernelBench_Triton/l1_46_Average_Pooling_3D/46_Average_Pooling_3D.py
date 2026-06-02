import triton
import triton.language as tl

@triton.jit
def avgpool3d_kernel(
    x_ptr, y_ptr,
    N, C, D, H, W,
    OD, OH, OW,
    SD, SH, SW,
    PD, PH, PW,
    n_elements,
    KSIZE_D: tl.constexpr, KSIZE_H: tl.constexpr, KSIZE_W: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask_o = offs < n_elements

    # Unravel linear index into (n, c, od, oh, ow)
    ow = offs % OW
    t = offs // OW
    oh = t % OH
    t = t // OH
    od = t % OD
    t = t // OD
    c = t % C
    n = t // C

    # Input start indices for each output element
    id_base = od * SD - PD
    ih_base = oh * SH - PH
    iw_base = ow * SW - PW

    # Accumulator in float32
    acc = tl.zeros([BLOCK], dtype=tl.float32)

    # Flattened base for nc
    base_nc = n * C + c
    base_ncD = base_nc * D

    # Iterate kernel window using unrolled static ranges
    for kd in tl.static_range(KSIZE_D):
        idv = id_base + kd
        md = (idv >= 0) & (idv < D)
        for kh in tl.static_range(KSIZE_H):
            ihv = ih_base + kh
            mh = (ihv >= 0) & (ihv < H)

            m_dh = mask_o & md & mh

            # Compute row base pointer once per (kd, kh)
            row_base = ((base_ncD + idv) * H + ihv) * W
            ptr_row = x_ptr + row_base

            # Start pointer for kw loop; increment by +1 each iteration (contiguous in W)
            p = ptr_row + iw_base

            # Generic unrolled kw loop
            for kw in tl.static_range(KSIZE_W):
                iwv = iw_base + kw
                mw = (iwv >= 0) & (iwv < W)
                m = m_dh & mw
                v = tl.load(p, mask=m, other=0.0)
                acc += v.to(tl.float32)
                p += 1

    # Average: divide by full kernel volume (count_include_pad=True)
    scale = 1.0 / float(KSIZE_D * KSIZE_H * KSIZE_W)
    out = acc * scale

    tl.store(y_ptr + offs, out, mask=mask_o)
