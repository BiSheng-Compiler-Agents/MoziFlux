import triton
import triton.language as tl

    @triton.jit
    def _lse_hswish_bias_clamp_kernel(
        x_ptr,             # *const float
        out_ptr,           # *float
        min_bias_ptr,      # *const float (1 element)
        M,                 # int32: total number of elements per (N*D*H*W)
        STRIDE_C,          # int32: stride between channels (D*H*W)
        C: tl.constexpr,   # number of channels to reduce over (compile-time constant)
        BLOCK: tl.constexpr,  # block size along M
    ):
        pid = tl.program_id(axis=0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < M

        min_bias = tl.load(min_bias_ptr).to(tl.float32)
        neg_inf = -float("inf")
        base_ptr = x_ptr + offs

        m = tl.full((BLOCK,), neg_inf, dtype=tl.float32)
        for c in tl.static_range(0, C):
            v = tl.load(base_ptr + c * STRIDE_C, mask=mask, other=neg_inf).to(tl.float32)
            m = tl.maximum(m, v)
        m = tl.where(mask, m, 0.0)

        s = tl.zeros((BLOCK,), dtype=tl.float32)
        for c in tl.static_range(0, C):
            v = tl.load(base_ptr + c * STRIDE_C, mask=mask, other=neg_inf).to(tl.float32)
            s += tl.exp(v - m)

        lse = tl.where(mask, m + tl.log(s), 0.0)
        t = lse + 3.0
        sig = 1.0 / (1.0 + tl.exp(-t))
        h = lse * sig * (1.0 / 6.0)
        z = h - min_bias
        z = tl.maximum(tl.minimum(z, 1.0), -1.0)
        tl.store(out_ptr + offs, z, mask=mask)
