import triton
import triton.language as tl


@triton.jit
def _gelu_groupnorm_kernel(
    x_ptr,  # *f32
    w_ptr,  # *f32
    b_ptr,  # *f32
    y_ptr,  # *f32
    N,
    C,
    H,
    W,  # i32
    G,  # i32
    eps,  # f32
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)  # each program handles one (n, g) group
    n = pid // G
    g = pid % G

    HW = H * W
    cpg = C // G  # channels per group
    group_elems = cpg * HW  # total elements in the group

    # Base offsets in flattened NCHW memory (contiguous)
    c_start = g * cpg
    n_base = n * C * H * W
    group_base = n_base + c_start * H * W

    offs = tl.arange(0, BLOCK)
    inv_sqrt2 = 0.7071067811865476

    # First pass: compute mean and variance over GELU(x) for this (n, g)
    x_group_ptr = x_ptr + group_base

    # Accumulate per-lane across the whole group, then reduce once to scalars
    acc1 = tl.zeros([BLOCK], dtype=tl.float32)
    acc2 = tl.zeros([BLOCK], dtype=tl.float32)

    idx = 0
    while idx < group_elems:
        i = idx + offs
        mask = i < group_elems

        # Load a contiguous slice of the group's data
        x = tl.load(x_group_ptr + i, mask=mask, other=0.0).to(tl.float32)

        # Exact GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
        z = 0.5 * x * (1.0 + tl.erf(x * inv_sqrt2))

        # Accumulate per-lane
        z = tl.where(mask, z, 0.0)
        acc1 += z
        acc2 += z * z

        idx += BLOCK

    s1 = tl.sum(acc1, axis=0)
    s2 = tl.sum(acc2, axis=0)

    ge_f = tl.full((), group_elems, dtype=tl.float32)
    mean = s1 / ge_f
    var = s2 / ge_f - mean * mean
    rstd = tl.rsqrt(var + eps)

    # Second pass: normalize and apply affine
    y_group_ptr = y_ptr + group_base

    ch = 0
    while ch < cpg:
        ch_base = ch * HW

        # Load affine params once per channel to avoid redundant gathers
        gamma = tl.load(w_ptr + c_start + ch).to(tl.float32)
        beta = tl.load(b_ptr + c_start + ch).to(tl.float32)

        off = 0
        while off < HW:
            idx_hw = off + offs
            mask_hw = idx_hw < HW

            x = tl.load(x_group_ptr + ch_base + idx_hw,
                        mask=mask_hw,
                        other=0.0).to(tl.float32)
            z = 0.5 * x * (1.0 + tl.erf(x * inv_sqrt2))

            y = (z - mean) * rstd
            y = y * gamma + beta

            tl.store(y_group_ptr + ch_base + idx_hw, y, mask=mask_hw)
            off += BLOCK
        ch += 1
