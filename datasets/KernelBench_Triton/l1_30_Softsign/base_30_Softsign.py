import torch
import torch.nn as nn

import triton
import triton.language as tl


@triton.jit
def _softsign_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    BLOCKS_PER_PROGRAM: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    base_block = pid * BLOCKS_PER_PROGRAM
    one = tl.full([1], 1.0, dtype=tl.float32)
    for chunk_idx in range(BLOCKS_PER_PROGRAM):
        block_idx = base_block + chunk_idx
        offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        x = tl.load(x_ptr + offsets, mask=mask, other=0)
        x_fp32 = x.to(tl.float32)
        y = x_fp32 / (tl.abs(x_fp32) + one)
        tl.store(y_ptr + offsets, y.to(x.dtype), mask=mask)


class ModelNew(nn.Module):

    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.device.type != "npu":
            raise RuntimeError("ModelNew expects input tensors on Ascend NPU")
        if x.dtype not in (torch.float16, torch.float32, torch.bfloat16):
            raise TypeError(f"Unsupported dtype for ModelNew: {x.dtype}")

        x_contiguous = x.contiguous()
        y = torch.empty_like(x_contiguous)
        n_elements = x_contiguous.numel()
        block_size = 8192
        blocks_per_program = 9
        num_blocks = triton.cdiv(n_elements, block_size)
        grid = (triton.cdiv(num_blocks, blocks_per_program), )

        _softsign_kernel[grid](
            x_contiguous.reshape(-1),
            y.reshape(-1),
            n_elements,
            BLOCK_SIZE=block_size,
            BLOCKS_PER_PROGRAM=blocks_per_program,
        )
        return y.reshape_as(x)


batch_size = 4096
dim = 393216


def get_inputs():
    x = torch.rand(batch_size, dim, device="npu")
    return [x]


def get_init_inputs():
    return []
