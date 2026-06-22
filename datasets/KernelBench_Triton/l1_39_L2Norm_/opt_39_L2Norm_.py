import torch
import torch.nn as nn
import torch_npu  # noqa: F401
import triton
import triton.language as tl

_MAX_PROGRAMS = 65535
_SMALL_MAX_D = 8192
_TILE_N = 4096


def _next_power_of_2(x: int) -> int:
    return 1 << (x - 1).bit_length()


@triton.jit
def _l2norm_small_kernel(x_ptr, y_ptr, M, N, sxm, sxn, sym, syn, n_programs,
                         BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    while row < M:
        mask = cols < N
        x = tl.load(x_ptr + row * sxm + cols * sxn, mask=mask,
                    other=0.0).to(tl.float32)
        ss = tl.sum(x * x, axis=0)
        inv = tl.rsqrt(ss)
        y = x * inv
        tl.store(y_ptr + row * sym + cols * syn, y, mask=mask)
        row += n_programs


@triton.jit
def _l2norm_large_rowwise_kernel(x_ptr, y_ptr, M, N, n_programs,
                                 BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    while row < M:
        acc = tl.zeros([1], dtype=tl.float32)
        start = 0
        while start < N:
            c = start + cols
            mask = c < N
            x = tl.load(x_ptr + row * N + c,
                        mask=mask,
                        other=0.0,
                        care_padding=False).to(tl.float32)
            acc += tl.sum(x * x, axis=0)
            start += BLOCK_N
        inv = tl.rsqrt(acc)
        start = 0
        while start < N:
            c = start + cols
            mask = c < N
            x = tl.load(x_ptr + row * N + c,
                        mask=mask,
                        other=0.0,
                        care_padding=False).to(tl.float32)
            tl.store(y_ptr + row * N + c, x * inv, mask=mask)
            start += BLOCK_N
        row += n_programs


class ModelNew(nn.Module):
    """Row-wise L2 normalization for a 2D tensor: y[i, :] = x[i, :] / ||x[i, :]||_2."""

    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not getattr(x, "is_npu", False):
            raise RuntimeError(
                "ModelNew expects an input tensor on Ascend NPU")
        if x.dim() != 2:
            raise ValueError("ModelNew expects a 2D input tensor")
        if x.requires_grad:
            raise RuntimeError(
                "ModelNew does not support autograd-enabled inputs")

        x_c = x.contiguous()
        M, N = x_c.shape
        y = torch.empty_like(x_c)
        if M == 0 or N == 0:
            return y

        if N <= _SMALL_MAX_D:
            block = _next_power_of_2(N)
            n_programs = min(M, _MAX_PROGRAMS)
            _l2norm_small_kernel[(n_programs, )](
                x_c,
                y,
                M,
                N,
                x_c.stride(0),
                x_c.stride(1),
                y.stride(0),
                y.stride(1),
                n_programs,
                BLOCK_N=block,
                num_warps=4,
                num_stages=2,
            )
            return y

        n_programs = min(M, _MAX_PROGRAMS)
        _l2norm_large_rowwise_kernel[(n_programs, )](
            x_c,
            y,
            M,
            N,
            n_programs,
            BLOCK_N=_TILE_N,
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
