import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 2048}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 4096}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 8192}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 16384}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 16384}, num_warps=16, num_stages=4),
    ],
    key=["M", "GROUP_SIZE"],
)
@triton.jit
def _group_mean_contrib_kernel(
        x_ptr,  # *f32 [N, C, D, H, W], contiguous in last 3 dims (per-channel)
        gamma_ptr,  # *f32 [C]
        sum_gamma_ptr,  # *f32 [G]
        out_ptr,  # *f32 [N] partial numerator contributions per sample
        N,
        C,
        G,
        M,  # ints
        stride_n,  # x.stride(0)
        stride_c,  # x.stride(1) == M for NCDHW contiguous
        eps,  # float32 epsilon
        GROUP_SIZE: tl.constexpr,  # channels per group
        BLOCK_M: tl.constexpr,  # tile over flattened spatial M = D*H*W
):
    pid = tl.program_id(axis=0)
    n = pid // G
    g = pid % G

    base_n = n * stride_n
    c_start = g * GROUP_SIZE

    # Accumulators in fp32
    acc_A = tl.zeros((), dtype=tl.float32)  # sum over group of x
    acc_B = tl.zeros((), dtype=tl.float32)  # sum over group of x^2
    acc_T = tl.zeros(
        (),
        dtype=tl.float32)  # sum over group of gamma[c] * sum_spatial(x_{n,c})

    offs = tl.arange(0, BLOCK_M)
    tl.max_contiguous(offs, BLOCK_M)

    # Loop over channels in the group; unrolled at compile-time
    for c_rel in tl.static_range(0, GROUP_SIZE):
        c = c_start + c_rel
        base = x_ptr + base_n + c * stride_c

        s_chan = tl.zeros((), dtype=tl.float32)
        ss_chan = tl.zeros((), dtype=tl.float32)

        m = 0
        while m < M:
            idx = m + offs
            mask = idx < M
            vals = tl.load(base + idx,
                           mask=mask,
                           other=0.0,
                           cache_modifier=".cg").to(tl.float32)
            s_chan += tl.sum(vals, axis=0)
            ss_chan += tl.sum(vals * vals, axis=0)
            m += BLOCK_M

        gamma_c = tl.load(gamma_ptr + c)
        acc_A += s_chan
        acc_B += ss_chan
        acc_T += gamma_c * s_chan

    # Group statistics
    Mg = GROUP_SIZE * M
    Mg_f32 = tl.full((), Mg, dtype=tl.float32)
    M_f32 = tl.full((), M, dtype=tl.float32)
    mu = acc_A / Mg_f32
    var = acc_B / Mg_f32 - mu * mu
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Contribution of this (n,g) to the numerator
    sum_gamma_g = tl.load(sum_gamma_ptr + g)
    term_g = (acc_T - mu * (M_f32 * sum_gamma_g)) * inv_std

    # Accumulate into out[n]
    tl.atomic_add(out_ptr + n, term_g)
