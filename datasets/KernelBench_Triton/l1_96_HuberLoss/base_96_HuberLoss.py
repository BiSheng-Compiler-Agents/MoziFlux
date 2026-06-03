import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _smooth_l1_mean_atomic_kernel(
    pred_ptr, tgt_ptr, out_mean_ptr,
    n_elements,
    inv_n,
    beta: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE

    CHUNK: tl.constexpr = 4096
    half_inv_beta: tl.constexpr = 0.5 / beta
    half_beta: tl.constexpr = 0.5 * beta
    acc = tl.zeros((), dtype=tl.float32)

    for off in range(0, BLOCK_SIZE, CHUNK):
        offsets = block_start + off + tl.arange(0, CHUNK)
        mask = offsets < n_elements

        tl.multiple_of(offsets, CHUNK)
        tl.max_contiguous(offsets, CHUNK)

        p = tl.load(pred_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        t = tl.load(tgt_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        d = p - t
        ad = tl.abs(d)

        small = (d * d) * half_inv_beta
        large = ad - half_beta
        loss = tl.where(ad < beta, small, large)
        loss = tl.where(mask, loss, 0.0)

        acc += tl.sum(loss, axis=0)

    tl.atomic_add(out_mean_ptr, acc * inv_n)


def smooth_l1_loss_triton(predictions: torch.Tensor, targets: torch.Tensor, beta: float = 1.0):
    assert predictions.shape == targets.shape, "predictions and targets must have the same shape"
    assert hasattr(torch, "npu"), "torch.npu is required for this operator"
    assert predictions.device.type == "npu", "predictions must be on NPU"
    assert targets.device.type == "npu", "targets must be on NPU"
    assert predictions.numel() > 0, "predictions must be non-empty"
    assert beta > 0, "beta must be positive"

    preds = predictions.contiguous()
    tgts = targets.contiguous()
    device = preds.device
    n_elements = preds.numel()

    out_mean = torch.zeros(1, device=device, dtype=torch.float32)
    inv_n = 1.0 / n_elements

    BLOCK_SIZE = 262144
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    _smooth_l1_mean_atomic_kernel[grid](
        preds, tgts, out_mean,
        n_elements,
        inv_n,
        beta,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
        num_stages=2,
    )
    return out_mean[0].to(predictions.dtype)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        return smooth_l1_loss_triton(predictions, targets, beta=1.0)


batch_size = 32768
input_shape = (32768,)
dim = 1


def get_inputs():
    scale = torch.rand(())
    return [torch.rand(batch_size, *input_shape) * scale, torch.rand(batch_size, *input_shape)]


def get_init_inputs():
    return []
