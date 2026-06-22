import torch
import torch.nn as nn
import torch_npu  # noqa: F401
import triton
import triton.language as tl

# PyTorch SELU constants
_SELU_SCALE = 1.0507009873554805
_SELU_ALPHA = 1.6732632423543772


@triton.jit
def _selu_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    grid_size,
    ALPHA: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    TILES_PER_PROGRAM: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    base_offsets = tl.arange(0, BLOCK_SIZE)
    alpha_scale = SCALE * ALPHA

    for tile_idx in range(TILES_PER_PROGRAM):
        tile_id = pid + tile_idx * grid_size
        offsets = tile_id * BLOCK_SIZE + base_offsets
        offsets = tl.max_contiguous(tl.multiple_of(offsets, 16), BLOCK_SIZE)

        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets,
                    mask=mask,
                    other=0.0,
                    eviction_policy="evict_first")
        x32 = x.to(tl.float32)

        exp_term = tl.exp(x32) - 1.0
        out32 = tl.where(x32 > 0, x32 * SCALE, exp_term * alpha_scale)
        tl.store(y_ptr + offsets, out32.to(x.dtype), mask=mask)


class ModelNew(nn.Module):

    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        supported_dtypes = {torch.float16, torch.bfloat16, torch.float32}
        if x.device.type != "npu":
            raise RuntimeError("ModelNew expects inputs on Ascend NPU")
        if x.dtype not in supported_dtypes:
            raise RuntimeError(f"Unsupported dtype for ModelNew: {x.dtype}")
        if x.requires_grad:
            raise RuntimeError(
                "ModelNew does not support autograd-tracked inputs")
        if x.numel() == 0:
            return x.contiguous()

        x_contig = x.contiguous()
        y = torch.empty_like(x_contig)
        n_elements = x_contig.numel()
        block_size = 4096
        total_tiles = triton.cdiv(n_elements, block_size)
        grid_size = min(total_tiles, 65535)
        tiles_per_program = triton.cdiv(total_tiles, grid_size)
        grid = (grid_size, )

        _selu_kernel[grid](
            x_contig,
            y,
            n_elements,
            grid_size,
            ALPHA=_SELU_ALPHA,
            SCALE=_SELU_SCALE,
            BLOCK_SIZE=block_size,
            TILES_PER_PROGRAM=tiles_per_program,
            num_warps=8,
            num_stages=2,
        )
        return y


batch_size = 4096
dim = 393216


def get_inputs():
    x = torch.rand(batch_size, dim)
    return [x]


def get_init_inputs():
    return []
