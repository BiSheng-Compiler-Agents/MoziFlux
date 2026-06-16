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
    def _hinge_loss_partial_kernel(pred_ptr, targ_ptr, partial_ptr):
        pid = tl.program_id(axis=0)
        offsets = tl.arange(0, 1024)
        acc = 0.0
        for chunk in range(pid * 1024, 32768, 32768):
            chunk_offsets = offsets + chunk
            chunk_offsets = tl.max_contiguous(
                tl.multiple_of(chunk_offsets, 1024), 1024)

            p = tl.load(pred_ptr + chunk_offsets)
            t = tl.load(targ_ptr + chunk_offsets)
            z = 1.0 - p * t
            acc += tl.sum(tl.maximum(z, 0.0), axis=0)
        tl.store(partial_ptr + pid, acc)

    @triton.jit
    def _hinge_loss_sum_kernel(partial_ptr, out_ptr):
        offsets = tl.arange(0, 32)
        vals = tl.load(partial_ptr + offsets)
        tl.store(out_ptr, tl.sum(vals, axis=0) * (1.0 / 32768.0))

    @triton.jit
    def _hinge_loss_sum_generic_kernel(pred_ptr, targ_ptr, out_ptr, n_elements,
                                       BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

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

        sum_buf = torch.empty(1, device=p.device, dtype=torch.float32)
        partial_buf = torch.empty(32, device=p.device, dtype=torch.float32)

        if N == 32768:
            _hinge_loss_partial_kernel[(32, )](p, t, partial_buf)
            _hinge_loss_sum_kernel[(1, )](partial_buf, sum_buf)
            return sum_buf[0]

        sum_buf = torch.zeros(1, device=p.device, dtype=torch.float32)

        def next_pow2(x: int) -> int:
            return 1 if x <= 1 else 1 << (x - 1).bit_length()

        BLOCK_SIZE = min(4096, max(256, next_pow2(N)))
        grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]), )
        _hinge_loss_sum_generic_kernel[grid](p,
                                             t,
                                             sum_buf,
                                             N,
                                             BLOCK_SIZE=BLOCK_SIZE)
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
