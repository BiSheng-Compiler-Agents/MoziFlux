import triton
import triton.language as tl

    @triton.jit
    def _perm_flip_w_kernel(
        inp_ptr,  # *T [CI, CO, KD, KH, KW]
        out_ptr,  # *T [CO, CI, KD, KH, KW]
        CI: tl.constexpr,
        CO: tl.constexpr,
        KD: tl.constexpr,
        KH: tl.constexpr,
        KW: tl.constexpr,
        N_ELEMS: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N_ELEMS

        # Decompose linear index into (co, ci, kd, kh, kw) for output layout [CO, CI, KD, KH, KW]
        kw = offs % KW
        tmp = offs // KW
        kh = tmp % KH
        tmp = tmp // KH
        kd = tmp % KD
        tmp = tmp // KD
        ci = tmp % CI
        co = tmp // CI

        # Map to input indices [CI, CO, KD, KH, KW] with spatial flip
        kd_in = KD - 1 - kd
        kh_in = KH - 1 - kh
        kw_in = KW - 1 - kw

        # Compute flat offsets for contiguous memory
        in_offs = (((ci * CO + co) * KD + kd_in) * KH + kh_in) * KW + kw_in
        # Output offset is simply offs (same linearization as we decomposed)
        out_offs = offs

        vals = tl.load(inp_ptr + in_offs, mask=mask, other=0)
        tl.store(out_ptr + out_offs, vals, mask=mask)

    @triton.jit
    def _perm_flip_w_kernel_grouped(
        inp_ptr,  # *T [CI, COG, KD, KH, KW]
        out_ptr,  # *T [CO, CIG, KD, KH, KW]
        CI: tl.constexpr,
        CO: tl.constexpr,
        KD: tl.constexpr,
        KH: tl.constexpr,
        KW: tl.constexpr,
        G: tl.constexpr,        # groups
        N_ELEMS: tl.constexpr,  # total elements of output
        BLOCK: tl.constexpr,
    ):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N_ELEMS

        kw = offs % KW
        t1 = offs // KW
        kh = t1 % KH
        t2 = t1 // KH
        kd = t2 % KD
        t3 = t2 // KD

        CIG = CI // G
        COG = CO // G

        ci_in_group = t3 % CIG
        co_global = t3 // CIG
        g = co_global // COG
        co_group = co_global - g * COG
        ci_global = g * CIG + ci_in_group

        kd_in = KD - 1 - kd
        kh_in = KH - 1 - kh
        kw_in = KW - 1 - kw

        # input is [CI, COG, KD, KH, KW]
        in_offs = (((ci_global * COG + co_group) * KD + kd_in) * KH + kh_in) * KW + kw_in
        out_offs = offs

        vals = tl.load(inp_ptr + in_offs, mask=mask, other=0)
        tl.store(out_ptr + out_offs, vals, mask=mask)
