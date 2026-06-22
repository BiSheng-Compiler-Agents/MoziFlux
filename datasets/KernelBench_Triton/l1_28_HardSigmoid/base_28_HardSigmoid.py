import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _hardsigmoid_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    BLOCKS_PER_PROGRAM: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    base = pid * BLOCK_SIZE * BLOCKS_PER_PROGRAM
    block_offsets = tl.arange(0, BLOCK_SIZE)

    for block_idx in range(BLOCKS_PER_PROGRAM):
        offsets = base + block_idx * BLOCK_SIZE + block_offsets
        mask = offsets < n_elements
        ptrs = x_ptr + offsets
        out_ptrs = y_ptr + offsets
        x = tl.load(ptrs, mask=mask, other=0.0)
        y_mid = x * (1.0 / 6.0) + 0.5
        y = tl.where(x <= -3.0, 0.0, y_mid)
        y = tl.where(x >= 3.0, 1.0, y)
        tl.store(out_ptrs, y, mask=mask)


class ModelNew(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.device.type != "npu":
            raise ValueError(
                f"ModelNew expects an Ascend NPU tensor, got device={x.device!s}"
            )
        if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise TypeError(f"Unsupported dtype for ModelNew: {x.dtype}")

        x_contig = x.contiguous()
        y = torch.empty_like(x_contig)
        n_elements = x_contig.numel()

        def grid(meta):
            return (triton.cdiv(
                n_elements, meta["BLOCK_SIZE"] * meta["BLOCKS_PER_PROGRAM"]), )

        _hardsigmoid_kernel[grid](
            x_contig,
            y,
            n_elements,
            BLOCK_SIZE=4096,
            BLOCKS_PER_PROGRAM=12,
            num_warps=4,
            num_stages=1,
        )
        return y


batch_size = 4096
dim = 393216


def get_inputs():
    x = torch.rand(batch_size, dim)
    return [x]


def get_init_inputs():
    return []
