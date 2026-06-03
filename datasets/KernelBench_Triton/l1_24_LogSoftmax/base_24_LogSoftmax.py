import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _log_softmax_streaming_kernel(
    x_ptr,
    y_ptr,
    B,
    D,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < B

    offs_n = tl.arange(0, BLOCK_N)

    running_max = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    running_sum = tl.zeros((BLOCK_M,), tl.float32)

    for start_n in range(0, D, BLOCK_N):
        cols = start_n + offs_n
        mask = row_mask[:, None] & (cols[None, :] < D)
        x_ptrs = x_ptr + rows[:, None] * D + cols[None, :]
        x = tl.load(x_ptrs, mask=mask, other=-float("inf"))
        x32 = x.to(tl.float32)
        tile_max = tl.max(x32, axis=1)
        new_max = tl.maximum(running_max, tile_max)
        running_sum = running_sum * tl.exp(running_max - new_max)
        running_sum += tl.sum(tl.exp(x32 - new_max[:, None]), axis=1)
        running_max = new_max

    log_denom = tl.log(running_sum)

    for start_n in range(0, D, BLOCK_N):
        cols = start_n + offs_n
        mask = row_mask[:, None] & (cols[None, :] < D)
        x_ptrs = x_ptr + rows[:, None] * D + cols[None, :]
        y_ptrs = y_ptr + rows[:, None] * D + cols[None, :]
        x = tl.load(x_ptrs, mask=mask, other=-float("inf"))
        y = x.to(tl.float32) - running_max[:, None] - log_denom[:, None]
        tl.store(y_ptrs, y, mask=mask)


@triton.jit
def _log_softmax_streaming_kernel_aligned(
    x_ptr,
    y_ptr,
    D,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    running_max = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    running_sum = tl.zeros((BLOCK_M,), tl.float32)
    row_bases = rows[:, None] * D

    for start_n in range(0, D, BLOCK_N):
        cols = start_n + offs_n
        x_ptrs = x_ptr + row_bases + cols[None, :]
        x = tl.load(x_ptrs)
        x32 = x.to(tl.float32)
        tile_max = tl.max(x32, axis=1)
        new_max = tl.maximum(running_max, tile_max)
        running_sum = running_sum * tl.exp(running_max - new_max)
        running_sum += tl.sum(tl.exp(x32 - new_max[:, None]), axis=1)
        running_max = new_max

    log_denom = tl.log(running_sum)

    for start_n in range(0, D, BLOCK_N):
        cols = start_n + offs_n
        x_ptrs = x_ptr + row_bases + cols[None, :]
        y_ptrs = y_ptr + row_bases + cols[None, :]
        x = tl.load(x_ptrs)
        y = x.to(tl.float32) - running_max[:, None] - log_denom[:, None]
        tl.store(y_ptrs, y)


def _pick_block_n(d: int) -> int:
    if d >= 262144:
        return 8064
    if d >= 65536:
        return 2048
    if d >= 8192:
        return 1024
    return 512


def _pick_num_warps(d: int) -> int:
    if d >= 262144:
        return 16
    if d >= 65536:
        return 8
    return 4


def _pick_num_stages(d: int) -> int:
    if d >= 262144:
        return 2
    return 2


def _pick_block_m(b: int, d: int) -> int:
    if b >= 2048 and d >= 262144:
        return 2
    if b >= 512 and d >= 65536:
        return 2
    return 1


def _log_softmax_triton(x: torch.Tensor, dim: int) -> torch.Tensor:
    if x.device.type != "npu":
        raise ValueError("log_softmax Triton path requires an Ascend NPU tensor.")
    if x.ndim != 2:
        raise ValueError("log_softmax Triton path expects a 2D tensor.")
    if dim not in (1, -1):
        raise ValueError("log_softmax Triton path supports only the last dimension.")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("Unsupported dtype for Triton log_softmax.")

    B, D = x.shape
    if B == 0 or D == 0:
        raise ValueError("log_softmax Triton path does not support empty tensors.")

    x_contig = x.contiguous()
    y = torch.empty_like(x_contig)

    block_m = _pick_block_m(B, D)
    block_n = _pick_block_n(D)
    num_warps = _pick_num_warps(D)
    num_stages = _pick_num_stages(D)

    if B % block_m == 0 and D % block_n == 0:
        _log_softmax_streaming_kernel_aligned[(B // block_m,)](
            x_contig,
            y,
            D,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    else:
        _log_softmax_streaming_kernel[(triton.cdiv(B, block_m),)](
            x_contig,
            y,
            B,
            D,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    return y


class ModelNew(nn.Module):
    def __init__(self, dim: int = 1):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _log_softmax_triton(x, self.dim)


batch_size = 4096
dim = 393216


def get_inputs():
    x = torch.rand(batch_size, dim)
    return [x]


def get_init_inputs():
    return []
