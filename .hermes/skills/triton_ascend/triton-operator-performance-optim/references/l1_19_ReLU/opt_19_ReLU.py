import torch
import torch.nn as nn
import torch_npu  # noqa: F401
import triton
import triton.language as tl


@triton.jit
def _relu_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    n_programs,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    tile_id = pid

    while tile_id * BLOCK_SIZE < n_elements:
        offsets = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        tl.multiple_of(offsets, 16)
        tl.max_contiguous(offsets, 16)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask, other=0)

        zero = tl.zeros([BLOCK_SIZE], dtype=x.dtype)
        y = tl.maximum(x, zero, propagate_nan=tl.PropagateNan.ALL)
        tl.store(y_ptr + offsets, y, mask=mask)
        tile_id += n_programs


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.device.type != "npu":
            raise ValueError("ModelNew expects an Ascend NPU tensor input")
        if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise TypeError(
                "ModelNew supports only float16, bfloat16, and float32 tensors"
            )
        if x.numel() == 0:
            return torch.empty_like(x)

        x_contig = x.contiguous()
        y = torch.empty_like(x_contig)
        x_flat = x_contig.view(-1)
        y_flat = y.view(-1)
        n_elements = x_flat.numel()
        block_size = 4096
        max_programs = 65535
        n_programs = min(triton.cdiv(n_elements, block_size), max_programs)
        grid = (n_programs,)
        _relu_kernel[grid](
            x_flat,
            y_flat,
            n_elements,
            n_programs,
            BLOCK_SIZE=block_size,
        )
        return y.view_as(x)


batch_size = 4096
dim = 393216


def get_inputs():
    x = torch.rand(batch_size, dim)
    return [x]


def get_init_inputs():
    return []
