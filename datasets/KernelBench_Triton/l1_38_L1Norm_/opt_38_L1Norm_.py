import torch
import torch.nn as nn
import torch_npu  # noqa: F401
import triton
import triton.language as tl

_MAX_PROGRAMS = 65535


@triton.jit
def _l1norm_row_kernel_opt(
    x_ptr,
    y_ptr,
    B,
    N,
    n_programs,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    cols = tl.arange(0, BLOCK_SIZE)
    tl.max_contiguous(cols, BLOCK_SIZE)
    tl.multiple_of(cols, 16)

    row = pid
    while row < B:
        row_base = row * N
        acc = tl.zeros((1, ), dtype=tl.float32)

        start = 0
        while start < N:
            offs0 = start + cols
            tl.multiple_of(offs0, 16)
            mask0 = offs0 < N
            x0 = tl.load(x_ptr + row_base + offs0,
                         mask=mask0,
                         other=0.0,
                         care_padding=False)
            acc += tl.sum(tl.abs(x0.to(tl.float32)), axis=0, keep_dims=True)

            offs1 = offs0 + BLOCK_SIZE
            tl.multiple_of(offs1, 16)
            mask1 = offs1 < N
            x1 = tl.load(x_ptr + row_base + offs1,
                         mask=mask1,
                         other=0.0,
                         care_padding=False)
            acc += tl.sum(tl.abs(x1.to(tl.float32)), axis=0, keep_dims=True)

            offs2 = offs1 + BLOCK_SIZE
            tl.multiple_of(offs2, 16)
            mask2 = offs2 < N
            x2 = tl.load(x_ptr + row_base + offs2,
                         mask=mask2,
                         other=0.0,
                         care_padding=False)
            acc += tl.sum(tl.abs(x2.to(tl.float32)), axis=0, keep_dims=True)

            offs3 = offs2 + BLOCK_SIZE
            tl.multiple_of(offs3, 16)
            mask3 = offs3 < N
            x3 = tl.load(x_ptr + row_base + offs3,
                         mask=mask3,
                         other=0.0,
                         care_padding=False)
            acc += tl.sum(tl.abs(x3.to(tl.float32)), axis=0, keep_dims=True)
            start += 4 * BLOCK_SIZE

        inv = 1.0 / acc

        start = 0
        while start < N:
            offs0 = start + cols
            tl.multiple_of(offs0, 16)
            mask0 = offs0 < N
            x0 = tl.load(x_ptr + row_base + offs0,
                         mask=mask0,
                         other=0.0,
                         care_padding=False).to(tl.float32)
            tl.store(y_ptr + row_base + offs0, x0 * inv, mask=mask0)

            offs1 = offs0 + BLOCK_SIZE
            tl.multiple_of(offs1, 16)
            mask1 = offs1 < N
            x1 = tl.load(x_ptr + row_base + offs1,
                         mask=mask1,
                         other=0.0,
                         care_padding=False).to(tl.float32)
            tl.store(y_ptr + row_base + offs1, x1 * inv, mask=mask1)

            offs2 = offs1 + BLOCK_SIZE
            tl.multiple_of(offs2, 16)
            mask2 = offs2 < N
            x2 = tl.load(x_ptr + row_base + offs2,
                         mask=mask2,
                         other=0.0,
                         care_padding=False).to(tl.float32)
            tl.store(y_ptr + row_base + offs2, x2 * inv, mask=mask2)

            offs3 = offs2 + BLOCK_SIZE
            tl.multiple_of(offs3, 16)
            mask3 = offs3 < N
            x3 = tl.load(x_ptr + row_base + offs3,
                         mask=mask3,
                         other=0.0,
                         care_padding=False).to(tl.float32)
            tl.store(y_ptr + row_base + offs3, x3 * inv, mask=mask3)
            start += 4 * BLOCK_SIZE

        row += n_programs


class ModelNew(nn.Module):
    """Row-wise L1 normalization: y = x / sum(abs(x), dim=1, keepdim=True)."""

    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.device.type != "npu":
            raise ValueError("ModelNew expects an Ascend NPU tensor")
        if x.ndim != 2:
            raise ValueError("ModelNew expects a 2D tensor")
        if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise TypeError(
                "ModelNew supports float16, bfloat16, and float32 inputs")

        x_contig = x.contiguous()
        B, N = x_contig.shape
        y = torch.empty_like(x_contig)
        if B == 0 or N == 0:
            return y

        if N >= 16384:
            block_size = 4096
        elif N >= 8192:
            block_size = 2048
        elif N >= 2048:
            block_size = 1024
        else:
            block_size = max(64, 1 << (N.bit_length() - 1))

        n_programs = min(B, _MAX_PROGRAMS)
        _l1norm_row_kernel_opt[(n_programs, )](
            x_contig,
            y,
            B,
            N,
            n_programs,
            BLOCK_SIZE=block_size,
            num_warps=4,
            num_stages=2,
        )
        return y


batch_size = 32768
dim = 65535


def get_inputs():
    x = torch.rand(batch_size, dim)
    return [x]


def get_init_inputs():
    return []
