import torch
import torch.nn as nn

import triton
import triton.language as tl


@triton.jit
def _maxpool3d_fwd_kernel(
    x_ptr,
    y_ptr,
    N,
    C,
    D,
    H,
    W,
    outD,
    outH,
    outW,
    stride_d,
    stride_h,
    stride_w,
    pad_d,
    pad_h,
    pad_w,
    dil_d,
    dil_h,
    dil_w,
    K_D: tl.constexpr,
    K_H: tl.constexpr,
    K_W: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    ROW_BLOCKS_PER_PROGRAM: tl.constexpr,
    BLOCK_W: tl.constexpr,
    USE_INT64_INDEX: tl.constexpr,
    SINGLE_W_TILE: tl.constexpr,
    TARGET_FASTPATH: tl.constexpr,
):
    pid_row_block = tl.program_id(0)
    if SINGLE_W_TILE:
        ow = tl.arange(0, BLOCK_W)
    else:
        pid_col = tl.program_id(1)
        ow = pid_col * BLOCK_W + tl.arange(0, BLOCK_W)
    if TARGET_FASTPATH:
        out_w_limit = 62
        out_h_limit = 62
        out_d_limit = 62
    else:
        out_w_limit = outW
        out_h_limit = outH
        out_d_limit = outD
    mask_ow = ow < out_w_limit
    tl.max_contiguous(ow, BLOCK_W)
    tl.multiple_of(ow, 1)

    if USE_INT64_INDEX:
        total_rows = tl.full((), N * C * outD * outH, dtype=tl.int64)
        row_block_start = tl.full(
            (),
            pid_row_block * BLOCK_ROWS * ROW_BLOCKS_PER_PROGRAM,
            dtype=tl.int64)
        row_offsets = tl.cast(tl.arange(0, BLOCK_ROWS), tl.int64)
        ow_index = tl.cast(ow, tl.int64)[None, :]
        if TARGET_FASTPATH:
            in_x0 = tl.cast(ow * 2 - 1, tl.int64)
            dhw = tl.full((), 128 * 128 * 128, dtype=tl.int64)
            out_hw = tl.full((), 62 * 62, dtype=tl.int64)
            l1 = tl.full((), 128 * 128, dtype=tl.int64)
            l2 = tl.full((), 128, dtype=tl.int64)
        else:
            in_x0 = tl.cast(ow * stride_w - pad_w, tl.int64)
            dhw = tl.full((), D * H * W, dtype=tl.int64)
            out_hw = tl.full((), outD * outH, dtype=tl.int64)
            l1 = tl.full((), H * W, dtype=tl.int64)
            l2 = tl.full((), W, dtype=tl.int64)
    else:
        total_rows = N * C * outD * outH
        row_block_start = pid_row_block * BLOCK_ROWS * ROW_BLOCKS_PER_PROGRAM
        row_offsets = tl.arange(0, BLOCK_ROWS)
        ow_index = ow[None, :]
        if TARGET_FASTPATH:
            in_x0 = ow * 2 - 1
            dhw = 128 * 128 * 128
            out_hw = 62 * 62
            l1 = 128 * 128
            l2 = 128
        else:
            in_x0 = ow * stride_w - pad_w
            dhw = D * H * W
            out_hw = outD * outH
            l1 = H * W
            l2 = W

    row_mask_ow = mask_ow[None, :]
    neg_inf = -float("inf")
    zero_f = 0.0
    if TARGET_FASTPATH:
        w_fp = 128.0
        h_fp = 128.0
        d_fp = 128.0
    else:
        w_fp = tl.cast(W, tl.float32)
        h_fp = tl.cast(H, tl.float32)
        d_fp = tl.cast(D, tl.float32)

    for row_block_idx in tl.static_range(0, ROW_BLOCKS_PER_PROGRAM):
        rows = row_block_start + row_block_idx * BLOCK_ROWS + row_offsets
        row_mask = rows < total_rows

        oh = rows % out_h_limit
        t = rows // out_h_limit
        od = t % out_d_limit
        t = t // out_d_limit
        c = t % C
        n = t // C

        if USE_INT64_INDEX:
            nc_index = tl.cast(n * C + c, tl.int64)
            od_index = tl.cast(od, tl.int64)
            oh_index = tl.cast(oh, tl.int64)
        else:
            nc_index = n * C + c
            od_index = od
            oh_index = oh

        if TARGET_FASTPATH:
            in_z0 = od_index[:, None] * 2 - 1
            in_y0 = oh_index[:, None] * 2 - 1
        else:
            in_z0 = od_index[:, None] * stride_d - pad_d
            in_y0 = oh_index[:, None] * stride_h - pad_h
        base_nc = nc_index[:, None] * dhw
        out_idx_base = (
            (nc_index * out_hw + od_index * out_h_limit + oh_index) *
            out_w_limit)[:, None]

        acc = tl.full((BLOCK_ROWS, BLOCK_W), neg_inf, dtype=tl.float32)
        x = in_x0[None, :]
        for kw in tl.static_range(0, K_W):
            x_fp = tl.cast(x, tl.float32)
            x_valid = (x_fp >= zero_f) & (x_fp < w_fp)
            if USE_INT64_INDEX:
                x_safe = tl.cast(tl.where(x_valid, x, 0), tl.int64)
            else:
                x_safe = tl.where(x_valid, x, 0)
            for kd in tl.static_range(0, K_D):
                z = in_z0 + kd * dil_d
                z_fp = tl.cast(z, tl.float32)
                z_valid = (z_fp >= zero_f) & (z_fp < d_fp)
                if USE_INT64_INDEX:
                    z_safe = tl.cast(tl.where(z_valid, z, 0), tl.int64)
                else:
                    z_safe = tl.where(z_valid, z, 0)
                z_base = z_safe * l1
                for kh in tl.static_range(0, K_H):
                    y = in_y0 + kh * dil_h
                    y_fp = tl.cast(y, tl.float32)
                    y_valid = (y_fp >= zero_f) & (y_fp < h_fp)
                    if USE_INT64_INDEX:
                        y_safe = tl.cast(tl.where(y_valid, y, 0), tl.int64)
                    else:
                        y_safe = tl.where(y_valid, y, 0)
                    base_zh = z_base + y_safe * l2
                    mask = row_mask[:,
                                    None] & row_mask_ow & x_valid & z_valid & y_valid
                    in_idx = base_nc + base_zh + x_safe
                    vals = tl.load(
                        x_ptr + in_idx,
                        mask=mask,
                        other=neg_inf,
                        eviction_policy="evict_first",
                    )
                    acc = tl.maximum(acc, vals.to(tl.float32))
            x += dil_w

        out_idx = out_idx_base + ow_index
        tl.store(y_ptr + out_idx, acc, mask=row_mask[:, None] & row_mask_ow)


def _as_triple(v):
    if isinstance(v, (tuple, list)):
        assert len(v) == 3
        return int(v[0]), int(v[1]), int(v[2])
    v = int(v)
    return (v, v, v)


def _compute_out_dim(in_size: int, k: int, stride: int, pad: int, dil: int,
                     ceil_mode: bool) -> int:
    eff = dil * (k - 1) + 1
    if ceil_mode:
        return max(0, (in_size + 2 * pad - eff + stride) // stride)
    return max(0, (in_size + 2 * pad - eff) // stride + 1)


class ModelNew(nn.Module):

    def __init__(self,
                 kernel_size: int,
                 stride: int = None,
                 padding: int = 0,
                 dilation: int = 1,
                 return_indices: bool = False,
                 ceil_mode: bool = False):
        super(ModelNew, self).__init__()
        if stride is None:
            stride = kernel_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.return_indices:
            raise NotImplementedError(
                "return_indices=True is not supported by the Triton implementation"
            )
        if self.ceil_mode:
            raise NotImplementedError(
                "ceil_mode=True is not supported by the Triton implementation")
        if not x.is_contiguous():
            raise ValueError("expected a contiguous input tensor")
        if x.dtype not in (torch.float16, torch.float32):
            raise TypeError(
                f"expected float16 or float32 input, got {x.dtype}")
        if x.device.type != "npu":
            raise ValueError(f"expected an NPU tensor, got device={x.device}")

        N, C, D, H, W = x.shape
        kD, kH, kW = _as_triple(self.kernel_size)
        sD, sH, sW = _as_triple(self.stride)
        pD, pH, pW = _as_triple(self.padding)
        dD, dH, dW = _as_triple(self.dilation)
        outD = _compute_out_dim(D, kD, sD, pD, dD, self.ceil_mode)
        outH = _compute_out_dim(H, kH, sH, pH, dH, self.ceil_mode)
        outW = _compute_out_dim(W, kW, sW, pW, dW, self.ceil_mode)
        if outD == 0 or outH == 0 or outW == 0:
            return x.new_empty((N, C, outD, outH, outW))

        y = torch.empty((N, C, outD, outH, outW),
                        device=x.device,
                        dtype=x.dtype)
        block_rows = 11
        row_blocks_per_program = 1
        block_w = 64
        total_rows = N * C * outD * outH
        max_input_offset = N * C * D * H * W - 1
        max_output_offset = N * C * outD * outH * outW - 1
        use_int64_index = max(max_input_offset, max_output_offset) >= 2**31
        single_w_tile = outW <= block_w
        target_fastpath = (not use_int64_index and D == 128 and H == 128
                           and W == 128 and outD == 62 and outH == 62
                           and outW == 62 and kD == 3 and kH == 3 and kW == 3
                           and sD == 2 and sH == 2 and sW == 2 and pD == 1
                           and pH == 1 and pW == 1 and dD == 3 and dH == 3
                           and dW == 3)
        if single_w_tile:
            grid = (triton.cdiv(total_rows,
                                block_rows * row_blocks_per_program), )
        else:
            grid = (
                triton.cdiv(total_rows, block_rows * row_blocks_per_program),
                triton.cdiv(outW, block_w),
            )
        _maxpool3d_fwd_kernel[grid](
            x,
            y,
            N,
            C,
            D,
            H,
            W,
            outD,
            outH,
            outW,
            sD,
            sH,
            sW,
            pD,
            pH,
            pW,
            dD,
            dH,
            dW,
            K_D=kD,
            K_H=kH,
            K_W=kW,
            BLOCK_ROWS=block_rows,
            ROW_BLOCKS_PER_PROGRAM=row_blocks_per_program,
            BLOCK_W=block_w,
            USE_INT64_INDEX=use_int64_index,
            SINGLE_W_TILE=single_w_tile,
            TARGET_FASTPATH=target_fastpath,
            num_warps=4,
            num_stages=2,
        )
        return y


def max_pool3d(
    x: torch.Tensor,
    kernel_size_: int | None = None,
    stride_: int | None = None,
    padding_: int | None = None,
    dilation_: int | None = None,
    return_indices: bool = False,
    ceil_mode: bool = False,
) -> torch.Tensor:
    if kernel_size_ is None:
        kernel_size_ = kernel_size
    if stride_ is None:
        stride_ = stride
    if padding_ is None:
        padding_ = padding
    if dilation_ is None:
        dilation_ = dilation
    return ModelNew(
        kernel_size=kernel_size_,
        stride=stride_,
        padding=padding_,
        dilation=dilation_,
        return_indices=return_indices,
        ceil_mode=ceil_mode,
    )(x)


batch_size = 16
channels = 32
dim1 = 128
dim2 = 128
dim3 = 128
kernel_size = 3
stride = 2
padding = 1
dilation = 3


def get_inputs():
    x = torch.rand(batch_size, channels, dim1, dim2, dim3)
    return [x]


def get_init_inputs():
    return [kernel_size, stride, padding, dilation]
