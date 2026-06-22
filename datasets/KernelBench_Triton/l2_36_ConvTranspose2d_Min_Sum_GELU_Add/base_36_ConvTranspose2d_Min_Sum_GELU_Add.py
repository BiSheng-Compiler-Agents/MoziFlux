import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _reduce_min_sum_gelu(
    x_ptr,
    tmp_ptr,
    N: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    sxn,
    sxc,
    sxh,
    sxw,
    stn,
    stw,
    BLOCK_W: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_w_blk = tl.program_id(1)

    w_offsets = pid_w_blk * BLOCK_W + tl.arange(0, BLOCK_W)
    mask_w = w_offsets < W

    x_base_n = pid_n * sxn

    w_ptrs = w_offsets * sxw
    tl.multiple_of(w_ptrs, values=1)

    acc = tl.zeros((BLOCK_W, ), dtype=tl.float32)
    inf = 1.0e20

    h = 0
    while (h + 3) < H:
        min0 = tl.full((BLOCK_W, ), inf, dtype=tl.float32)
        min1 = tl.full((BLOCK_W, ), inf, dtype=tl.float32)
        min2 = tl.full((BLOCK_W, ), inf, dtype=tl.float32)
        min3 = tl.full((BLOCK_W, ), inf, dtype=tl.float32)

        c_start = 0
        while c_start < C:
            c_tile = c_start + tl.arange(0, BLOCK_C)
            mask_ct = c_tile < C
            base_cw = x_ptr + x_base_n + c_tile[:,
                                                None] * sxc + w_ptrs[None, :]
            mask_cw = mask_ct[:, None] & mask_w[None, :]

            v0 = tl.load(
                base_cw + (h + 0) * sxh,
                mask=mask_cw,
                other=inf,
                cache_modifier=".cg",
            ).to(tl.float32)
            v1 = tl.load(
                base_cw + (h + 1) * sxh,
                mask=mask_cw,
                other=inf,
                cache_modifier=".cg",
            ).to(tl.float32)
            v2 = tl.load(
                base_cw + (h + 2) * sxh,
                mask=mask_cw,
                other=inf,
                cache_modifier=".cg",
            ).to(tl.float32)
            v3 = tl.load(
                base_cw + (h + 3) * sxh,
                mask=mask_cw,
                other=inf,
                cache_modifier=".cg",
            ).to(tl.float32)

            min0 = tl.minimum(min0, tl.min(v0, axis=0))
            min1 = tl.minimum(min1, tl.min(v1, axis=0))
            min2 = tl.minimum(min2, tl.min(v2, axis=0))
            min3 = tl.minimum(min3, tl.min(v3, axis=0))
            c_start += BLOCK_C

        acc += tl.where(mask_w, (min0 + min1) + (min2 + min3), 0.0)
        h += 4

    while h < H:
        cur_min = tl.full((BLOCK_W, ), inf, dtype=tl.float32)
        c_start = 0
        while c_start < C:
            c_tile = c_start + tl.arange(0, BLOCK_C)
            mask_ct = c_tile < C
            base_cw = x_ptr + x_base_n + c_tile[:,
                                                None] * sxc + w_ptrs[None, :]
            x_vals = tl.load(
                base_cw + h * sxh,
                mask=mask_ct[:, None] & mask_w[None, :],
                other=inf,
                cache_modifier=".cg",
            ).to(tl.float32)
            cur_min = tl.minimum(cur_min, tl.min(x_vals, axis=0))
            c_start += BLOCK_C
        acc += tl.where(mask_w, cur_min, 0.0)
        h += 1

    inv_sqrt2 = 0.7071067811865476
    gelu_vals = 0.5 * acc * (1.0 + tl.math.erf(acc * inv_sqrt2))
    tmp_ptrs = tmp_ptr + pid_n * stn + w_offsets * stw
    tl.store(tmp_ptrs, gelu_vals, mask=mask_w)


@triton.jit
def _broadcast_bias_add(
    tmp_ptr,
    bias_ptr,
    y_ptr,
    N: tl.constexpr,
    C: tl.constexpr,
    W: tl.constexpr,
    stn,
    stw,
    sbc,
    syn,
    syc,
    syw,
    BLOCK_W: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_w_blk = tl.program_id(1)
    pid_c_blk = tl.program_id(2)

    w_offsets = pid_w_blk * BLOCK_W + tl.arange(0, BLOCK_W)
    c_offsets = pid_c_blk * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_w = w_offsets < W
    mask_c = c_offsets < C

    tmp_vals = tl.load(tmp_ptr + pid_n * stn + w_offsets * stw,
                       mask=mask_w,
                       other=0.0).to(tl.float32)
    bias_vals = tl.load(bias_ptr + c_offsets * sbc, mask=mask_c,
                        other=0.0).to(tl.float32)
    out_ptrs = y_ptr + pid_n * syn + c_offsets[:, None] * syc + w_offsets[
        None, :] * syw
    out_tile = tmp_vals[None, :] + bias_vals[:, None]
    tl.store(out_ptrs, out_tile, mask=mask_c[:, None] & mask_w[None, :])


class ModelNew(nn.Module):

    def __init__(
            self,
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
            bias_shape=(128, 1, 1),
    ):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels,
                                                 kernel_size, stride, padding,
                                                 output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = self.conv_transpose(x)
        n_dim, c_dim, h_dim, w_dim = x.shape
        y = torch.empty((n_dim, c_dim, 1, w_dim),
                        device=x.device,
                        dtype=x.dtype)
        reduced = torch.empty((n_dim, w_dim),
                              device=x.device,
                              dtype=torch.float32)

        x_c = x.contiguous()
        b_c = self.bias.contiguous()

        block_w = 128
        block_c = 32
        reduce_grid = (n_dim, triton.cdiv(w_dim, block_w))
        store_grid = (n_dim, triton.cdiv(w_dim,
                                         block_w), triton.cdiv(c_dim, block_c))

        _reduce_min_sum_gelu[reduce_grid](
            x_c,
            reduced,
            n_dim,
            c_dim,
            h_dim,
            w_dim,
            x_c.stride(0),
            x_c.stride(1),
            x_c.stride(2),
            x_c.stride(3),
            reduced.stride(0),
            reduced.stride(1),
            BLOCK_W=block_w,
            BLOCK_C=block_c,
            num_warps=4,
            num_stages=2,
        )

        _broadcast_bias_add[store_grid](
            reduced,
            b_c,
            y,
            n_dim,
            c_dim,
            w_dim,
            reduced.stride(0),
            reduced.stride(1),
            b_c.stride(0),
            y.stride(0),
            y.stride(1),
            y.stride(3),
            BLOCK_W=block_w,
            BLOCK_C=block_c,
            num_warps=1,
            num_stages=2,
        )
        return y


batch_size = 16
in_channels = 64
out_channels = 128
height, width = 128, 128
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
bias_shape = (out_channels, 1, 1)


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [
        in_channels, out_channels, kernel_size, stride, padding,
        output_padding, bias_shape
    ]
