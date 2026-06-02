import triton
import triton.language as tl

@triton.jit
def _fused_mean_bias_lse(
    x_ptr,           # float32[N, C, H, W] - contiguous NCHW
    bias_ptr,        # float32[C, 1, 1]
    out_ptr,         # float32[N]
    N, C, H, W,
    stride_n, stride_c,
    bias_stride_c,
    BLOCK_C: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid
    if n >= N:
        return

    HW = H * W
    inv_hw = 1.0 / HW
    n_base = n * stride_n

    NEG_INF = -float("inf")
    m = tl.full((), NEG_INF, dtype=tl.float32)
    s = tl.zeros((), dtype=tl.float32)

    c_arange = tl.arange(0, BLOCK_C)

    for c_start in range(0, C, BLOCK_C):
        c_idx = c_start + c_arange
        c_mask = c_idx < C

        sum_c = tl.zeros((BLOCK_C,), dtype=tl.float32)

        base_c = n_base + c_idx * stride_c
        ptrs_base = base_c[:, None]

        hw_arange = tl.arange(0, BLOCK_HW)
        offs_hw = hw_arange
        hw_mask = offs_hw < HW
        ptrs = ptrs_base + offs_hw[None, :]
        load_mask = c_mask[:, None] & hw_mask[None, :]
        tile = tl.load(x_ptr + ptrs, mask=load_mask, other=0.0, cache_modifier=".cg")

        for hw_start in range(BLOCK_HW, HW, BLOCK_HW):
            sum_c += tl.sum(tile, axis=1)
            offs_hw = hw_start + hw_arange
            hw_mask = offs_hw < HW
            ptrs = ptrs_base + offs_hw[None, :]
            load_mask = c_mask[:, None] & hw_mask[None, :]
            tile = tl.load(x_ptr + ptrs, mask=load_mask, other=0.0, cache_modifier=".cg")

        sum_c += tl.sum(tile, axis=1)

        mean_c = sum_c * inv_hw
        b = tl.load(bias_ptr + c_idx * bias_stride_c, mask=c_mask, other=0.0, cache_modifier=".ca")
        v = mean_c + b
        v = tl.where(c_mask, v, NEG_INF)

        tile_max = tl.max(v, axis=0)
        m2 = tl.maximum(m, tile_max)
        s = s * tl.exp(m - m2) + tl.sum(tl.exp(v - m2), axis=0)
        m = m2

    lse = tl.log(s) + m
    tl.store(out_ptr + n, 10.0 * lse)
