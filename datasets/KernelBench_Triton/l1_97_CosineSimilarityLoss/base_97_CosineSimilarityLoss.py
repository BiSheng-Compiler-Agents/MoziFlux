import torch
import torch.nn as nn
import torch_npu  # noqa: F401
import triton
import triton.language as tl


@triton.jit
def _cosine_similarity_rows_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    B,
    D,
    EPS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < B
    cols = tl.arange(0, BLOCK_N)

    dot = tl.zeros((BLOCK_M, ), dtype=tl.float32)
    nx2 = tl.zeros((BLOCK_M, ), dtype=tl.float32)
    ny2 = tl.zeros((BLOCK_M, ), dtype=tl.float32)

    x_row_ptr = x_ptr + rows[:, None] * D
    y_row_ptr = y_ptr + rows[:, None] * D

    for start in range(0, D, BLOCK_N):
        offs = start + cols
        mask = row_mask[:, None] & (offs[None, :] < D)
        x = tl.load(x_row_ptr + offs[None, :], mask=mask, other=0.0)
        y = tl.load(y_row_ptr + offs[None, :], mask=mask, other=0.0)

        x32 = x.to(tl.float32)
        y32 = y.to(tl.float32)

        dot += tl.sum(x32 * y32, axis=1)
        nx2 += tl.sum(x32 * x32, axis=1)
        ny2 += tl.sum(y32 * y32, axis=1)

    denom = tl.sqrt(nx2) * tl.sqrt(ny2)
    denom = tl.maximum(denom, EPS, propagate_nan=tl.PropagateNan.ALL)
    tl.store(out_ptr + rows, 1.0 - (dot / denom), mask=row_mask)


@triton.jit
def _cosine_similarity_rows_kernel_full_tiles(
    x_ptr,
    y_ptr,
    out_ptr,
    D,
    EPS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_TILES: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)

    dot = tl.zeros((BLOCK_M, ), dtype=tl.float32)
    nx2 = tl.zeros((BLOCK_M, ), dtype=tl.float32)
    ny2 = tl.zeros((BLOCK_M, ), dtype=tl.float32)

    x_row_ptr = x_ptr + rows[:, None] * D
    y_row_ptr = y_ptr + rows[:, None] * D

    for tile_idx in range(0, NUM_TILES):
        offs = tile_idx * BLOCK_N + cols
        x32 = tl.load(x_row_ptr + offs[None, :]).to(tl.float32)
        y32 = tl.load(y_row_ptr + offs[None, :]).to(tl.float32)

        dot += tl.sum(x32 * y32, axis=1)
        nx2 += tl.sum(x32 * x32, axis=1)
        ny2 += tl.sum(y32 * y32, axis=1)

    denom = tl.sqrt(nx2) * tl.sqrt(ny2)
    denom = tl.maximum(denom, EPS, propagate_nan=tl.PropagateNan.ALL)
    tl.store(out_ptr + rows, 1.0 - (dot / denom))


class ModelNew(nn.Module):

    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        if predictions.ndim != 2 or targets.ndim != 2:
            raise ValueError(
                "ModelNew expects 2D predictions and targets tensors")
        if predictions.shape != targets.shape:
            raise ValueError(
                "predictions and targets must have the same shape")
        if not getattr(predictions, "is_npu", False) or not getattr(
                targets, "is_npu", False):
            raise RuntimeError(
                "ModelNew requires predictions and targets on Ascend NPU")

        x = predictions.contiguous()
        y = targets.contiguous()
        B, D = x.shape
        block_m = 4
        block_n = 2048

        out = torch.empty(B, device=x.device, dtype=torch.float32)

        if (B % block_m) == 0 and (D % block_n) == 0:
            grid = (B // block_m, )
            _cosine_similarity_rows_kernel_full_tiles[grid](
                x,
                y,
                out,
                D,
                EPS=1e-8,
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                NUM_TILES=D // block_n,
            )
        else:
            grid = (triton.cdiv(B, block_m), )
            _cosine_similarity_rows_kernel[grid](
                x,
                y,
                out,
                B,
                D,
                EPS=1e-8,
                BLOCK_M=block_m,
                BLOCK_N=block_n,
            )

        return out.mean()


batch_size = 128
input_shape = (4096, )
dim = 1


def get_inputs():
    return [
        torch.randn(batch_size, *input_shape),
        torch.randn(batch_size, *input_shape)
    ]


def get_init_inputs():
    return []
