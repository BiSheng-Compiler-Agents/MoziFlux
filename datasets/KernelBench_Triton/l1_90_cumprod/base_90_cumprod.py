import torch
import torch.nn as nn
import triton
import triton.language as tl


BLOCK_N = 512
NUM_WARPS = 4
NUM_STAGES = 2


@triton.jit
def _cumprod_rowwise_kernel_vectorized(
    x_ptr,
    y_ptr,
    M,
    N,
    stride_xm,
    stride_xn,
    stride_ym,
    stride_yn,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    if pid_m >= M:
        return

    cols = tl.arange(0, BLOCK_N)
    x_row_ptr = x_ptr + pid_m * stride_xm
    y_row_ptr = y_ptr + pid_m * stride_ym
    carry = 1.0

    i = 0
    while i < N:
        offsets = i + cols
        mask = offsets < N
        vals = tl.load(x_row_ptr + offsets * stride_xn, mask=mask, other=1.0)
        tile_prefix = tl.cumprod(vals, axis=0) * carry
        tl.store(y_row_ptr + offsets * stride_yn, tile_prefix, mask=mask)
        carry = tl.sum(tile_prefix * (cols == (BLOCK_N - 1)), axis=0)
        i += BLOCK_N


@triton.jit
def _touch_first_elem(y_ptr, M, stride_ym, stride_yn):
    pid = tl.program_id(axis=0)
    if pid >= M:
        return
    row_ptr = y_ptr + pid * stride_ym
    v = tl.load(row_ptr + 0 * stride_yn)
    tl.store(row_ptr + 0 * stride_yn, v)


class ModelNew(nn.Module):
    def __init__(self, dim=1):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x):
        if not x.is_npu:
            raise RuntimeError("ModelNew expects inputs on Ascend NPU")
        if x.ndim == 0:
            raise RuntimeError("cumprod requires at least one dimension")

        dim = self.dim
        if dim < 0:
            dim += x.ndim
        if dim < 0 or dim >= x.ndim:
            raise IndexError(f"dim={self.dim} is out of range for ndim={x.ndim}")

        if x.numel() == 0:
            return torch.empty_like(x)

        perm = [idx for idx in range(x.ndim) if idx != dim] + [dim]
        moved = x.permute(perm).contiguous()
        scan_len = moved.shape[-1]
        flat = moved.reshape(-1, scan_len)
        out_flat = torch.empty_like(flat)

        grid = (flat.shape[0],)
        _cumprod_rowwise_kernel_vectorized[grid](
            flat,
            out_flat,
            flat.shape[0],
            flat.shape[1],
            flat.stride(0),
            flat.stride(1),
            out_flat.stride(0),
            out_flat.stride(1),
            BLOCK_N=BLOCK_N,
            num_warps=NUM_WARPS,
            num_stages=NUM_STAGES,
        )

        out = out_flat.reshape(moved.shape)
        inv_perm = [0] * x.ndim
        for idx, src in enumerate(perm):
            inv_perm[src] = idx
        return out.permute(inv_perm)


batch_size = 32768
input_shape = (32768,)
dim = 1


def get_inputs():
    return [torch.rand(batch_size, *input_shape)]


def get_init_inputs():
    return [dim]
