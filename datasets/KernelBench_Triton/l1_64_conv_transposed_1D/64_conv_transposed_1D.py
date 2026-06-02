import triton
import triton.language as tl

@triton.jit
def _deconv1d_stride1_kernel(
    x_ptr,         # *f32/f16/bf16 [B, C_IN, L_IN]
    w_ptr,         # *f32/f16/bf16 [C_IN, C_OUT, K]
    b_ptr,         # *f32[C_OUT] or dummy if no bias
    y_ptr,         # *dtype(x)[B, C_OUT, L_OUT]
    B,
    C_IN: tl.constexpr,
    C_OUT,
    L_IN,
    K: tl.constexpr,
    L_OUT,
    BLOCK_T: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    pid0 = tl.program_id(axis=0)
    pid1 = tl.program_id(axis=1)

    batch_idx = pid0 // C_OUT
    oc_idx = pid0 % C_OUT
    t_offsets = pid1 * BLOCK_T + tl.arange(0, BLOCK_T)
    t_mask = t_offsets < L_OUT

    acc = tl.zeros((BLOCK_T,), dtype=tl.float32)

    x_batch_base = batch_idx * (C_IN * L_IN)
    y_base = batch_idx * (C_OUT * L_OUT) + oc_idx * L_OUT

    for cin_idx in tl.static_range(0, C_IN):
        x_base = x_batch_base + cin_idx * L_IN
        w_base = (cin_idx * C_OUT + oc_idx) * K
        for k_idx in tl.static_range(0, K):
            t_in = t_offsets - k_idx
            valid_t = (t_in >= 0) & (t_in < L_IN) & t_mask
            safe_t_in = tl.where(valid_t, t_in, 0)
            x_vals = tl.load(x_ptr + x_base + safe_t_in, mask=valid_t, other=0.0).to(tl.float32)
            w_val = tl.load(w_ptr + w_base + k_idx).to(tl.float32)
            acc += w_val * x_vals

    if HAS_BIAS:
        acc += tl.load(b_ptr + oc_idx).to(tl.float32)

    tl.store(y_ptr + y_base + t_offsets, acc, mask=t_mask)
