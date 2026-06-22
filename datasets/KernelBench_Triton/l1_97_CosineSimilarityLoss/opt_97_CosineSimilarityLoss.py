import torch
import torch.nn as nn
import torch_npu  # noqa: F401
import triton
import triton.language as tl

_MAX_PROGRAMS = 65535


@triton.jit
def _cosine_similarity_rows_contig_direct_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    B,
    D,
    EPS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < D
    base = pid * D + offs
    x = tl.load(x_ptr + base, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(y_ptr + base, mask=mask, other=0.0).to(tl.float32)
    dot = tl.sum(x * y, axis=0)
    nx2 = tl.sum(x * x, axis=0)
    ny2 = tl.sum(y * y, axis=0)
    denom = tl.maximum(tl.sqrt(nx2) * tl.sqrt(ny2), EPS)
    loss = 1.0 - (dot / denom)
    tl.store(out_ptr + pid, loss, mask=pid < B)


@triton.jit
def _cosine_similarity_rows_contig_persistent_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    B,
    D,
    n_programs,
    EPS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs = tl.arange(0, BLOCK_SIZE)
    row = pid
    while row < B:
        mask = offs < D
        base = row * D + offs
        x = tl.load(x_ptr + base, mask=mask, other=0.0).to(tl.float32)
        y = tl.load(y_ptr + base, mask=mask, other=0.0).to(tl.float32)
        dot = tl.sum(x * y, axis=0)
        nx2 = tl.sum(x * x, axis=0)
        ny2 = tl.sum(y * y, axis=0)
        denom = tl.maximum(tl.sqrt(nx2) * tl.sqrt(ny2), EPS)
        loss = 1.0 - (dot / denom)
        tl.store(out_ptr + row, loss, mask=row < B)
        row += n_programs


class ModelNew(nn.Module):
    """Cosine similarity loss: mean(1 - cosine_similarity(predictions, targets, dim=1))."""

    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        if predictions.ndim != 2 or targets.ndim != 2:
            raise ValueError(
                "ModelNew expects 2D predictions and targets tensors")
        if predictions.shape != targets.shape:
            raise ValueError(
                "predictions and targets must have the same shape")
        if predictions.device != targets.device:
            raise RuntimeError(
                "predictions and targets must be on the same device")
        if not getattr(predictions, "is_npu", False) or not getattr(
                targets, "is_npu", False):
            raise RuntimeError(
                "ModelNew requires predictions and targets on Ascend NPU")

        x = predictions.contiguous()
        y = targets.contiguous()
        B, D = x.shape
        out = torch.empty((B, ), device=x.device, dtype=torch.float32)
        block_size = max(1, triton.next_power_of_2(D))
        if B <= _MAX_PROGRAMS:
            _cosine_similarity_rows_contig_direct_kernel[(B, )](
                x,
                y,
                out,
                B,
                D,
                EPS=1e-8,
                BLOCK_SIZE=block_size,
            )
        else:
            n_programs = _MAX_PROGRAMS
            _cosine_similarity_rows_contig_persistent_kernel[(n_programs, )](
                x,
                y,
                out,
                B,
                D,
                n_programs,
                EPS=1e-8,
                BLOCK_SIZE=block_size,
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
