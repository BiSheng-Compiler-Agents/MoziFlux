import triton
import triton.language as tl

@triton.jit
def _relu_groupnorm_kernel(
    x_ptr,           # *T
    y_ptr,           # *T
    w_ptr,           # *fp32
    b_ptr,           # *fp32
    N, C, D, H, W,   # int32
    G,               # int32
    eps,             # fp32
    sC,              # int32 = D*H*W
    sN,              # int32 = C*sC
    GROUP_ELEMS,     # int32 = (C//G) * sC
    NUM_TILES: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // G
    g = pid % G

    group_channels = C // G
    c_start = g * group_channels
    base_group = n * sN + c_start * sC
    group_elems = GROUP_ELEMS

    offs = tl.arange(0, BLOCK_SIZE)

    # First pass: compute mean and var of ReLU(x) over the group (double-buffered prefetch)
    sum1 = tl.zeros([], dtype=tl.float32)
    sum2 = tl.zeros([], dtype=tl.float32)
    if NUM_TILES == 1:
        idx0 = offs
        mask0 = idx0 < group_elems
        x0 = tl.load(x_ptr + base_group + idx0, mask=mask0, other=0.0)
        z0 = tl.maximum(x0.to(tl.float32), 0.0)
        sum1 += tl.sum(z0, axis=0)
        sum2 += tl.sum(z0 * z0, axis=0)
    else:
        idx0 = offs
        mask0 = idx0 < group_elems
        x0 = tl.load(x_ptr + base_group + idx0, mask=mask0, other=0.0)
        for t in tl.static_range(1, NUM_TILES):
            idx1 = t * BLOCK_SIZE + offs
            mask1 = idx1 < group_elems
            x1 = tl.load(x_ptr + base_group + idx1, mask=mask1, other=0.0)

            z0 = tl.maximum(x0.to(tl.float32), 0.0)
            sum1 += tl.sum(z0, axis=0)
            sum2 += tl.sum(z0 * z0, axis=0)

            x0 = x1
            idx0 = idx1
            mask0 = mask1

        # tail
        z0 = tl.maximum(x0.to(tl.float32), 0.0)
        sum1 += tl.sum(z0, axis=0)
        sum2 += tl.sum(z0 * z0, axis=0)

    denom = tl.full([], group_elems, dtype=tl.float32)
    mean = sum1 / denom
    var = sum2 / denom - mean * mean
    inv_std = tl.rsqrt(var + eps)

    # Second pass: iterate per-channel to minimize gamma/beta global loads
    ci = 0
    while ci < group_channels:
        ch = c_start + ci
        gamma = tl.load(w_ptr + ch).to(tl.float32)
        beta = tl.load(b_ptr + ch).to(tl.float32)
        base_ch = n * sN + ch * sC

        off_c = 0
        while off_c < sC:
            idx = off_c + offs
            mask = idx < sC
            xr = tl.load(x_ptr + base_ch + idx, mask=mask, other=0.0)
            x_dtype = xr.dtype
            z = tl.maximum(xr.to(tl.float32), 0.0)
            z = (z - mean) * inv_std
            y = z * gamma + beta
            tl.store(y_ptr + base_ch + idx, y.to(x_dtype), mask=mask)
            off_c += BLOCK_SIZE
        ci += 1
