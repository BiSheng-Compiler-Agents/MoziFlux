import torch
import torch.nn as nn
import torch_npu  # noqa: F401
import triton
import triton.language as tl


def _is_npu_tensor(x: torch.Tensor) -> bool:
    return bool(getattr(x, "is_npu", False))


@triton.jit
def _argmax_row_kernel(
    x_ptr,
    out_ptr,
    b,
    k,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    rows = tl.max_contiguous(rows, BLOCK_M)
    row_mask = rows < b

    best_val = tl.full((BLOCK_M, ), -float("inf"), dtype=tl.float32)
    best_idx = tl.zeros((BLOCK_M, ), dtype=tl.int32)
    row_ptrs = x_ptr + rows[:, None] * k

    k0 = 0
    while k0 < k:
        cols = k0 + tl.arange(0, BLOCK_K)
        cols = tl.max_contiguous(cols, BLOCK_K)
        mask = row_mask[:, None] & (cols[None, :] < k)
        values = tl.load(
            row_ptrs + cols[None, :],
            mask=mask,
            other=-float("inf"),
            cache_modifier=".cg",
        ).to(tl.float32)

        tile_max = tl.max(values, axis=1)
        match_mask = (values == tile_max[:, None]) & mask
        invalid_index = tl.full((BLOCK_M, BLOCK_K), k, dtype=tl.int32)
        tile_indices = tl.where(match_mask, cols[None, :].to(tl.int32),
                                invalid_index)
        tile_first_idx = tl.min(tile_indices, axis=1)
        should_update = (tile_max > best_val) | ((tile_max == best_val) &
                                                 (tile_first_idx < best_idx))
        best_val = tl.where(should_update, tile_max, best_val)
        best_idx = tl.where(should_update, tile_first_idx, best_idx)
        k0 += BLOCK_K

    tl.store(out_ptr + rows, best_idx.to(tl.int64), mask=row_mask)


def argmax_over_a_dimension(x: torch.Tensor, dim: int) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        raise TypeError("argmax_over_a_dimension expects a torch.Tensor input")
    if not _is_npu_tensor(x):
        raise RuntimeError(
            "argmax_over_a_dimension expects an Ascend NPU tensor")
    if x.dim() == 0:
        raise ValueError(
            "argmax_over_a_dimension expects a tensor with rank at least 1")
    if x.dtype not in (torch.float16, torch.float32, torch.bfloat16):
        raise TypeError(
            "argmax_over_a_dimension supports only float16, float32, and bfloat16 inputs"
        )

    dim = int(dim)
    if dim < 0:
        dim += x.dim()
    if dim < 0 or dim >= x.dim():
        raise ValueError(
            f"invalid reduction dim {dim} for input rank {x.dim()}")
    if x.shape[dim] == 0:
        raise ValueError(
            "argmax_over_a_dimension does not support empty reduction axes")

    x_last = x.movedim(dim, -1).contiguous()
    rows = x_last.numel() // x_last.shape[-1]
    cols = x_last.shape[-1]
    x_2d = x_last.view(rows, cols)

    out = torch.empty((rows, ), device=x.device, dtype=torch.int64)
    grid = (triton.cdiv(rows, 8), )
    _argmax_row_kernel[grid](
        x_2d,
        out,
        rows,
        cols,
        BLOCK_M=8,
        BLOCK_K=1024,
        num_warps=4 if 1024 <= 256 else 8,
        num_stages=2,
    )

    out_shape = list(x.shape)
    del out_shape[dim]
    return out.view(*out_shape)


class ModelNew(nn.Module):

    def __init__(self, dim: int = 1):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return argmax_over_a_dimension(x, self.dim)


batch_size = 128
dim1 = 4096
dim2 = 4095


def get_inputs():
    x = torch.rand(batch_size, dim1, dim2)
    return [x]


def get_init_inputs():
    return [1]
