import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=4),
        triton.Config({}, num_warps=8, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=4),
        triton.Config({}, num_warps=16, num_stages=2),
    ],
    key=["S", "C"],
)
@triton.jit
def _fused_hswish_relu_softmax_mean_kernel(
    x_ptr,  # *float or *half, shape [N, C, S] (S = D*H*W), contiguous N,C,S layout
    out_ptr,  # *float or *half, shape [N, C]
    N,
    C,
    S,  # int32
    stride_n,  # int32, elements between successive n
    stride_c,  # int32, elements between successive c
    stride_out_n,  # int32, elements between successive n in out
    inv_S,  # float32, 1.0 / S
    BLOCK_C: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    pid_n = tl.program_id(axis=0)
    if pid_n >= N:
        return

    base_n = pid_n * stride_n
    c_idx = tl.arange(0, BLOCK_C)
    valid_c = c_idx < C

    # Accumulator over spatial positions for each channel
    acc = tl.zeros((BLOCK_C, ), dtype=tl.float32)

    inv6 = 1.0 / 6.0
    s_start = 0
    while s_start < S:
        s_idx = s_start + tl.arange(0, BLOCK_S)
        valid_s = s_idx < S
        mask_s = valid_s[None, :]

        # Offsets for 2D tile [C, S_tile]
        offs = base_n + c_idx[:, None] * stride_c + s_idx[None, :]

        # Load in fp32 for stability; mask only along S (C tile == C in launcher)
        x = tl.load(x_ptr + offs, mask=mask_s, other=0.0).to(tl.float32)

        # Fused HardSwish + ReLU: y = max(x, 0) * clamp(x + 3, 0, 6) / 6
        t = tl.minimum(tl.maximum(x + 3.0, 0.0), 6.0)
        y = tl.maximum(x, 0.0) * (t * inv6)

        # Mask invalid spatial lanes with -inf for softmax max-reduction
        y_masked = tl.where(mask_s, y, -float("inf"))

        # Softmax across channels (axis=0) per spatial column
        m = tl.max(y_masked, axis=0)  # [BLOCK_S]
        expv = tl.exp(y_masked - m[None, :])  # [BLOCK_C, BLOCK_S]
        sumexp = tl.sum(expv, axis=0)  # [BLOCK_S]
        p = expv / sumexp[None, :]  # [BLOCK_C, BLOCK_S]
        p = tl.where(mask_s, p, 0.0)

        # Accumulate probabilities across spatial positions for each channel
        acc += tl.sum(p, axis=1)

        s_start += BLOCK_S

    # Write normalized mean over spatial dims
    out_offs = pid_n * stride_out_n + c_idx
    tl.store(out_ptr + out_offs, acc * inv_S, mask=valid_c)
