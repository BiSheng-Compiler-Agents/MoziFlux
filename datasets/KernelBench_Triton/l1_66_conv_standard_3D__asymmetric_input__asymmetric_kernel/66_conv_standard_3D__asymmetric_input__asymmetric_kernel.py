import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=8, num_stages=3),
        # Tuned configs for Hopper-class GPUs (H200)
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 16}, num_warps=4, num_stages=3),
        # Extra high-occupancy configs for larger tiles on H200
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 128}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=8, num_stages=4),
    ],
    key=["M", "Cout", "K"],
)
@triton.jit
def _conv3d_implicit_gemm_kernel(
    x_ptr,       # *: [N, Cin, Din, Hin, Win] contiguous
    w2d_ptr,     # *: [K, Cout] contiguous, where K = Cin*Kd*Kh*Kw
    y_ptr,       # float32* [N, Cout, Dout, Hout, Wout] contiguous
    N, Cin, Din, Hin, Win,
    Cout, Kd, Kh, Kw,
    Dout, Hout, Wout,
    stride_d, stride_h, stride_w,
    pad_d, pad_h, pad_w,
    dil_d, dil_h, dil_w,
    M, K,  # M = N*Dout*Hout*Wout, K = Cin*Kd*Kh*Kw
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # rows: flattened (n, d, h, w)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # cols: output channels

    m_mask = offs_m < M
    n_mask = offs_n < Cout

    # map offs_m -> (n_idx, d_idx, h_idx, w_idx)
    dhw = Dout * Hout * Wout
    hw = Hout * Wout

    n_idx = offs_m // dhw
    rem = offs_m - n_idx * dhw
    d_idx = rem // hw
    rem = rem - d_idx * hw
    h_idx = rem // Wout
    w_idx = rem - h_idx * Wout

    # base input coords for this output location (with stride/padding)
    in_d_base = d_idx * stride_d - pad_d
    in_h_base = h_idx * stride_h - pad_h
    in_w_base = w_idx * stride_w - pad_w

    # accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Precompute strides for faster address arithmetic (NCDHW and NCDHW for output)
    sN_x = Cin * Din * Hin * Win
    sC_x = Din * Hin * Win
    sD_x = Hin * Win
    sH_x = Win

    sN_y = Cout * Dout * Hout * Wout
    sC_y = Dout * Hout * Wout
    sD_y = Hout * Wout
    sH_y = Wout

    # reduction over K
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K

        # map offs_k -> (ci, kd, kh, kw) given flatten order ((ci * Kd + kd) * Kh + kh) * Kw + kw
        kw_idx = offs_k % Kw
        tmp1 = offs_k // Kw
        kh_idx = tmp1 % Kh
        tmp2 = tmp1 // Kh
        kd_idx = tmp2 % Kd
        ci_idx = tmp2 // Kd

        # broadcast to (BM, BK)
        in_d = in_d_base[:, None] + kd_idx[None, :] * dil_d
        in_h = in_h_base[:, None] + kh_idx[None, :] * dil_h
        in_w = in_w_base[:, None] + kw_idx[None, :] * dil_w

        # bounds check on input coordinates
        valid_in = (
            m_mask[:, None]
            & k_mask[None, :]
            & (in_d >= 0)
            & (in_d < Din)
            & (in_h >= 0)
            & (in_h < Hin)
            & (in_w >= 0)
            & (in_w < Win)
        )

        # compute input addresses (NCDHW contiguous) using stride arithmetic
        addr_x = (
            n_idx[:, None] * sN_x
            + ci_idx[None, :] * sC_x
            + in_d * sD_x
            + in_h * sH_x
            + in_w
        )
        x_tile = tl.load(x_ptr + addr_x.to(tl.int64), mask=valid_in, other=0)

        # load weight tile w2d: [K, Cout]
        addr_w = offs_k[:, None] * Cout + offs_n[None, :]
        valid_w = k_mask[:, None] & n_mask[None, :]
        w_tile = tl.load(w2d_ptr + addr_w.to(tl.int64), mask=valid_w, other=0)

        # Mixed-precision friendly dot: if inputs are fp16, this will use Tensor Cores with fp32 accumulation.
        # If inputs are fp32, Triton will do fp32 math (TF32 may be used on Hopper depending on settings).
        acc += tl.dot(x_tile, w_tile)

    # store to y (NCDHW contiguous) via precomputed strides
    y_addr = (
        n_idx[:, None] * sN_y
        + offs_n[None, :] * sC_y
        + d_idx[:, None] * sD_y
        + h_idx[:, None] * sH_y
        + w_idx[:, None]
    )
    y_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(y_ptr + y_addr.to(tl.int64), acc, mask=y_mask)
