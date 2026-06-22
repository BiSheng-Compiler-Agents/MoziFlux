import torch
import torch.nn as nn
import torch_npu  # noqa: F401
import triton
import triton.language as tl


@triton.jit
def _leaky_relu_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    neg,
    BLOCK_SIZE: tl.constexpr,
    BLOCKS_PER_PROGRAM: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    lane_offsets = tl.arange(0, BLOCK_SIZE)
    tl.multiple_of(lane_offsets, 16)
    tl.max_contiguous(lane_offsets, 16)

    for block_idx in tl.static_range(0, BLOCKS_PER_PROGRAM):
        block_start = (pid * BLOCKS_PER_PROGRAM + block_idx) * BLOCK_SIZE
        offsets = block_start + lane_offsets
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask, other=0)
        y = x * tl.where(x >= 0, 1, neg)
        tl.store(y_ptr + offsets, y, mask=mask)


class ModelNew(nn.Module):

    def __init__(self, negative_slope: float = 0.01):
        super(ModelNew, self).__init__()
        self.negative_slope = negative_slope

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
        n_elements = x.numel()
        block_size = 7168
        blocks_per_program = 5
        grid = (triton.cdiv(n_elements, block_size * blocks_per_program), )
        y = torch.empty_like(x_contig)
        _leaky_relu_kernel[grid](
            x_contig.view(-1),
            y.view(-1),
            n_elements,
            float(self.negative_slope),
            BLOCK_SIZE=block_size,
            BLOCKS_PER_PROGRAM=blocks_per_program,
            num_warps=4,
            num_stages=2,
        )
        return y.view_as(x)


batch_size = 4096
dim = 393216


def get_inputs():
    x = torch.rand(batch_size, dim)
    return [x]


def get_init_inputs():
    return []
