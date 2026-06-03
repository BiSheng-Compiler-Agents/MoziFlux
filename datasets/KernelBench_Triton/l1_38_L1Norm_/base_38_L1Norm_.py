import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _l1norm_row_kernel(
    x_ptr, y_ptr,
    B, N,
    stride_xm, stride_xn,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    UNROLL: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    row_mask = rows[:, None] < B

    tl.max_contiguous(cols, BLOCK_N)
    tl.multiple_of(cols, 8)

    x_row_ptr = x_ptr + rows[:, None] * stride_xm
    y_row_ptr = y_ptr + rows[:, None] * stride_ym
    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
    start = 0
    while start < N:
        offs0 = start + cols
        tl.multiple_of(offs0, 8)
        mask0 = row_mask & (offs0[None, :] < N)
        x0 = tl.load(x_row_ptr + offs0[None, :] * stride_xn, mask=mask0, other=0.0, cache_modifier=".cg")
        acc += tl.sum(tl.abs(x0.to(tl.float32)), axis=1)

        if UNROLL > 1:
            offs1 = offs0 + BLOCK_N
            tl.multiple_of(offs1, 8)
            mask1 = row_mask & (offs1[None, :] < N)
            x1 = tl.load(
                x_row_ptr + offs1[None, :] * stride_xn,
                mask=mask1,
                other=0.0,
                cache_modifier=".cg",
            )
            acc += tl.sum(tl.abs(x1.to(tl.float32)), axis=1)

        start += UNROLL * BLOCK_N

    denom = acc
    inv = 1.0 / denom

    start = 0
    while start < N:
        offs0 = start + cols
        tl.multiple_of(offs0, 8)
        mask0 = row_mask & (offs0[None, :] < N)
        x0 = tl.load(
            x_row_ptr + offs0[None, :] * stride_xn,
            mask=mask0,
            other=0.0,
            cache_modifier=".cg",
        ).to(tl.float32)
        tl.store(y_row_ptr + offs0[None, :] * stride_yn, x0 * inv[:, None], mask=mask0, eviction_policy="evict_last")

        if UNROLL > 1:
            offs1 = offs0 + BLOCK_N
            tl.multiple_of(offs1, 8)
            mask1 = row_mask & (offs1[None, :] < N)
            x1 = tl.load(
                x_row_ptr + offs1[None, :] * stride_xn,
                mask=mask1,
                other=0.0,
                cache_modifier=".cg",
            ).to(tl.float32)
            tl.store(
                y_row_ptr + offs1[None, :] * stride_yn,
                x1 * inv[:, None],
                mask=mask1,
                eviction_policy="evict_last",
            )

        start += UNROLL * BLOCK_N


class ModelNew(nn.Module):
    """
    Performs row-wise L1 normalization with a Triton kernel on Ascend NPU.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.device.type != "npu":
            raise ValueError("ModelNew expects an Ascend NPU tensor")
        if x.ndim != 2:
            raise ValueError("ModelNew expects a 2D tensor")
        if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise TypeError("ModelNew supports float16, bfloat16, and float32 inputs")

        B, N = x.shape
        x_contig = x.contiguous()
        y = torch.empty_like(x_contig)

        stride_xm, stride_xn = x_contig.stride()
        stride_ym, stride_yn = y.stride()

        if N >= 32768:
            BLOCK_M = 8
            BLOCK_N = 2048
            UNROLL = 1
        elif N >= 8192:
            BLOCK_M = 4
            BLOCK_N = 1024
            UNROLL = 2
        elif N >= 2048:
            BLOCK_M = 4
            BLOCK_N = 512
            UNROLL = 2
        else:
            BLOCK_M = 1
            BLOCK_N = max(64, (1 << (N.bit_length() - 1)) if N > 0 else 1)
            UNROLL = 1

        BLOCK_M = min(BLOCK_M, max(B, 1))

        _l1norm_row_kernel[(triton.cdiv(B, BLOCK_M),)](
            x_contig, y,
            B, N,
            stride_xm, stride_xn,
            stride_ym, stride_yn,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            UNROLL=UNROLL,
        )
        return y
batch_size = 32768
dim = 65535

def get_inputs():
    x = torch.rand(batch_size, dim)
    return [x]
def get_init_inputs():
    return []
