import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_C": 16}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 256, "BLOCK_C": 16}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 512, "BLOCK_C": 16}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 256, "BLOCK_C": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 512, "BLOCK_C": 32}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 1024, "BLOCK_C": 16}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 1024, "BLOCK_C": 32}, num_warps=8, num_stages=3),
    ],
    key=["C"],
)
@triton.jit
def _softmax_sub_swish_max_kernel(
    x_ptr,            # *f32, input tensor [B, C, D, H, W]
    sub_ptr,          # *f32, subtract vector [C]
    out_ptr,          # *f32, output tensor [B, D, H, W]
    B, C, D, H, W,    # int32 sizes
    BLOCK_M: tl.constexpr,  # number of (n,d,h,w) positions per program
    BLOCK_C: tl.constexpr,  # channel tile
):
    pid = tl.program_id(axis=0)

    # Total number of spatial positions per sample and altogether
    sC = D * H * W         # elements per channel map (stride along channel)
    M = B * sC             # total positions across batch and spatial dims
    batch_span = C * sC    # number of elements per batch

    # Indices of positions this program handles
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = offs_m < M

    # Map linear position to (n, inner_m)
    n = offs_m // sC
    inner_m = offs_m % sC
    base = n * batch_span + inner_m  # base offset for channel 0 for each position

    neg_inf = -float('inf')

    # Fast path: if all channels fit into one tile, do everything in a single pass.
    if C <= BLOCK_C:
        c_idx = tl.arange(0, BLOCK_C)
        c_mask = c_idx < C
        offs = base[None, :] + c_idx[:, None] * sC
        load_mask = c_mask[:, None] & m_mask[None, :]
        x_vals = tl.load(x_ptr + offs, mask=load_mask, other=neg_inf, cache_modifier=".ca").to(tl.float32)

        # stable softmax
        m_row = tl.max(x_vals, axis=0)
        ex = tl.exp(x_vals - m_row[None, :])
        s = tl.sum(ex, axis=0)
        inv_s = 1.0 / s

        sub_vals = tl.load(sub_ptr + c_idx, mask=c_mask, other=0.0, cache_modifier=".ca").to(tl.float32)
        y = ex * inv_s[None, :]  # softmax
        z = y - sub_vals[:, None]

        sig = 1.0 / (1.0 + tl.exp(-z))
        s_val = sig * z
        s_val = tl.where(load_mask, s_val, neg_inf)
        out_max = tl.max(s_val, axis=0)
        tl.store(out_ptr + offs_m, out_max, mask=m_mask)
        return

    # Two-pass streaming softmax (general path)
    m = tl.full((BLOCK_M,), neg_inf, dtype=tl.float32)
    s = tl.zeros((BLOCK_M,), dtype=tl.float32)

    # Pass 1: compute per-(B,d,h,w) max and normalizer
    for c0 in range(0, C, BLOCK_C):
        c_idx = c0 + tl.arange(0, BLOCK_C)
        c_mask = c_idx < C
        offs = base[None, :] + c_idx[:, None] * sC
        load_mask = c_mask[:, None] & m_mask[None, :]
        x_vals = tl.load(x_ptr + offs, mask=load_mask, other=neg_inf, cache_modifier=".cg").to(tl.float32)
        tile_max = tl.max(x_vals, axis=0)
        m_new = tl.maximum(m, tile_max)
        scale = tl.exp(m - m_new)
        ex_block = tl.exp(x_vals - m_new[None, :])
        s = s * scale + tl.sum(ex_block, axis=0)
        m = m_new

    # Avoid NaNs on masked lanes
    m = tl.where(m_mask, m, 0.0)
    s = tl.where(m_mask, s, 1.0)
    inv_s = 1.0 / s

    # Pass 2: compute fused: softmax -> subtract -> swish; then max over channels
    out_max = tl.full((BLOCK_M,), neg_inf, dtype=tl.float32)
    for c0 in range(0, C, BLOCK_C):
        c_idx = c0 + tl.arange(0, BLOCK_C)
        c_mask = c_idx < C

        offs = base[None, :] + c_idx[:, None] * sC
        load_mask = c_mask[:, None] & m_mask[None, :]
        x_vals = tl.load(x_ptr + offs, mask=load_mask, other=neg_inf, cache_modifier=".cg").to(tl.float32)

        # softmax probs using computed m and inv_s
        y = tl.exp(x_vals - m[None, :]) * inv_s[None, :]

        # load subtract vector for this channel tile
        sub_vals = tl.load(sub_ptr + c_idx, mask=c_mask, other=0.0, cache_modifier=".ca").to(tl.float32)
        z = y - sub_vals[:, None]  # broadcast over positions

        # swish: sigmoid(z) * z
        sig = 1.0 / (1.0 + tl.exp(-z))
        s_val = sig * z

        # mask out invalid lanes to avoid affecting max
        s_val = tl.where(load_mask, s_val, neg_inf)
        tile_max = tl.max(s_val, axis=0)
        out_max = tl.maximum(out_max, tile_max)

    # Store output [B, D, H, W], which is linearized by offs_m
    tl.store(out_ptr + offs_m, out_max, mask=m_mask)
