import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except Exception:
    triton = None
    tl = None
    _TRITON_AVAILABLE = False

if _TRITON_AVAILABLE:

    @triton.jit
    def _hinge_loss_sum_kernel(pred_ptr, targ_ptr, out_ptr, n_elements,
                               BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        tl.multiple_of(block_start, BLOCK_SIZE)
        tl.max_contiguous(offsets, BLOCK_SIZE)

        p = tl.load(pred_ptr + offsets, mask=mask, other=1.0)
        t = tl.load(targ_ptr + offsets, mask=mask, other=1.0)
        z = 1.0 - p * t
        z = tl.maximum(z, 0.0)
        part = tl.sum(z, axis=0)

        if n_elements <= BLOCK_SIZE:
            if pid == 0:
                tl.store(out_ptr, part)
        else:
            tl.atomic_add(out_ptr, part)


class ModelNew(nn.Module):
    """
    A model that computes Hinge Loss for binary classification tasks.

    Parameters:
        None
    """

    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        if not _TRITON_AVAILABLE:
            raise RuntimeError("Triton is required for ModelNew")
        if predictions.device.type != "npu" or targets.device.type != "npu":
            raise RuntimeError("ModelNew expects NPU tensors")
        supported_dtypes = {torch.float16, torch.bfloat16, torch.float32}
        if predictions.dtype not in supported_dtypes or targets.dtype not in supported_dtypes:
            raise TypeError(
                "ModelNew expects float16, bfloat16, or float32 inputs")
        if predictions.requires_grad or targets.requires_grad:
            raise RuntimeError(
                "ModelNew does not support autograd-tracked tensors")
        if predictions.numel() != targets.numel():
            raise ValueError(
                "predictions and targets must have the same number of elements"
            )
        N = predictions.numel()
        if N == 0:
            raise ValueError("predictions and targets must be non-empty")

        p = predictions.contiguous().view(-1).to(torch.float32)
        t = targets.contiguous().view(-1).to(torch.float32)
        sum_buf = torch.zeros(1, device=p.device, dtype=torch.float32)

        def next_pow2(x: int) -> int:
            return 1 if x <= 1 else 1 << (x - 1).bit_length()

        BLOCK_SIZE = min(4096, max(256, next_pow2(N)))

        if BLOCK_SIZE >= 2048:
            num_warps, num_stages = 8, 2
        elif BLOCK_SIZE >= 1024:
            num_warps, num_stages = 4, 2
        elif BLOCK_SIZE >= 512:
            num_warps, num_stages = 2, 1
        else:
            num_warps, num_stages = 1, 1

        grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]), )
        _hinge_loss_sum_kernel[grid](p,
                                     t,
                                     sum_buf,
                                     N,
                                     BLOCK_SIZE=BLOCK_SIZE,
                                     num_warps=num_warps,
                                     num_stages=num_stages)

        return sum_buf[0] / N


batch_size = 32768
input_shape = (32768, )
dim = 1


def get_inputs():
    return [
        torch.rand(batch_size, *input_shape),
        torch.randint(0, 2, (batch_size, )).float() * 2 - 1
    ]


def get_init_inputs():
    return []
