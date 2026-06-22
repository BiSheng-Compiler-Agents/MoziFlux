import torch
import torch.nn as nn
import torch_npu  # noqa: F401
import triton
import triton.language as tl

TARGET_M = 65536
TARGET_N = 16384
TARGET_NUMEL = TARGET_M * TARGET_N
FALLBACK_BLOCK_SIZE = 32768


@triton.jit
def _scale_kernel(x_ptr, y_ptr, s, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    block_start = pid * BLOCK_SIZE
    stride = num_programs * BLOCK_SIZE
    while block_start < n_elements:
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        x = tl.load(x_ptr + offsets, cache_modifier=".cg")
        s_cast = tl.full((), s, x.dtype)
        tl.store(y_ptr + offsets, x * s_cast)
        block_start += stride


@triton.jit
def _scale_kernel_fallback(x_ptr, y_ptr, s, n_elements,
                           BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0, cache_modifier=".cg")
    s_cast = tl.full((), s, x.dtype)
    tl.store(y_ptr + offsets, x * s_cast, mask=mask)


class ModelNew(nn.Module):

    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A: torch.Tensor, s: float) -> torch.Tensor:
        if not getattr(A, "is_npu", False):
            raise ValueError("ModelNew expects input tensor A on Ascend NPU")
        if not A.is_contiguous():
            A = A.contiguous()
        C = torch.empty_like(A)
        n_elements = A.numel()
        if n_elements == 0:
            return C

        if A.ndim == 2 and A.shape == (TARGET_M,
                                       TARGET_N) and A.dtype == torch.float32:

            def grid(meta):
                exact_blocks = TARGET_NUMEL // meta["BLOCK_SIZE"]
                return (min(exact_blocks, 32768), )

            _scale_kernel[grid](A,
                                C,
                                float(s),
                                TARGET_NUMEL,
                                BLOCK_SIZE=16384,
                                num_warps=8,
                                num_stages=1)
            return C

        def grid(meta):
            return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )

        _scale_kernel_fallback[grid](A,
                                     C,
                                     float(s),
                                     n_elements,
                                     BLOCK_SIZE=FALLBACK_BLOCK_SIZE,
                                     num_warps=8,
                                     num_stages=1)
        return C


M = 16384 * 4
N = 4096 * 4


def get_inputs():
    A = torch.rand(M, N)
    s = 3.14
    return [A, s]


def get_init_inputs():
    return []
