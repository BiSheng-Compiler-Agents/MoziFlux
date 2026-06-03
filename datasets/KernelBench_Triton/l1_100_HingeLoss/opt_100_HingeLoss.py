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

# Number of partial reduction slots = AIV core count
# Each Phase-1 program writes to its own slot — zero atomic contention.
_NUM_PARTS = 32

if _TRITON_AVAILABLE:

    @triton.jit
    def _hinge_loss_partial_kernel(pred_ptr, targ_ptr, partial_ptr,
                                   N, BLOCK: tl.constexpr,
                                   NUM_PARTS: tl.constexpr):
        """Phase 1 — Strided partial reduction.

        NUM_PARTS programs stride through the input in parallel.
        Each program accumulates into its own partial slot.
        No tl.atomic_add — fully parallel write.
        """
        pid = tl.program_id(axis=0)
        acc = 0.0
        for chunk in range(pid * BLOCK, N, NUM_PARTS * BLOCK):
            offs = chunk + tl.arange(0, BLOCK)
            offs = tl.max_contiguous(tl.multiple_of(offs, BLOCK), BLOCK)
            mask = offs < N

            p = tl.load(pred_ptr + offs, mask=mask, other=1.0,
                        care_padding=False)
            t = tl.load(targ_ptr + offs, mask=mask, other=1.0,
                        care_padding=False)

            # Hinge loss: max(0, 1 - p * t)  (always computed in FP32)
            z = 1.0 - p * t
            z = tl.maximum(z, 0.0)
            acc += tl.sum(z, axis=0)

        tl.store(partial_ptr + pid, acc)

    @triton.jit
    def _hinge_loss_reduce_kernel(partial_ptr, out_ptr,
                                  N, NUM_PARTS: tl.constexpr):
        """Phase 2 — Single program sums partials and normalises by N."""
        vals = tl.load(partial_ptr + tl.arange(0, NUM_PARTS))
        tl.store(out_ptr, tl.sum(vals, axis=0) / N)

    @triton.jit
    def _hinge_loss_direct_kernel(pred_ptr, targ_ptr, out_ptr,
                                  N, BLOCK: tl.constexpr):
        """Direct fallback when N fits in one tile — no reduction needed."""
        pid = tl.program_id(axis=0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N

        p = tl.load(pred_ptr + offs, mask=mask, other=1.0,
                    care_padding=False)
        t = tl.load(targ_ptr + offs, mask=mask, other=1.0,
                    care_padding=False)

        z = 1.0 - p * t
        z = tl.maximum(z, 0.0)
        part = tl.sum(z, axis=0)

        if pid == 0:
            tl.store(out_ptr, part / N)


class ModelNew(nn.Module):
    """Optimised Hinge Loss with two-phase reduction.

    Eliminates tl.atomic_add serialisation via two-phase reduction:
      Phase 1: NUM_PARTS programs compute partial sums into private slots.
      Phase 2: one program sums the partials and normalises.
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
            raise TypeError("ModelNew expects float16, bfloat16, or float32 inputs")
        if predictions.requires_grad or targets.requires_grad:
            raise RuntimeError("ModelNew does not support autograd-tracked tensors")
        if predictions.numel() != targets.numel():
            raise ValueError("predictions and targets must have the same number of elements")
        N = predictions.numel()
        if N == 0:
            raise ValueError("predictions and targets must be non-empty")

        # Normalise to contiguous FP32 flat tensors
        p = predictions.contiguous().view(-1).to(torch.float32)
        t = targets.contiguous().view(-1).to(torch.float32)

        sum_buf = torch.empty(1, device=p.device, dtype=torch.float32)
        # Zero-initialise so idle programs contribute 0, not garbage
        partial_buf = torch.zeros(_NUM_PARTS, device=p.device, dtype=torch.float32)

        def _next_pow2(x: int) -> int:
            return 1 if x <= 1 else 1 << (x - 1).bit_length()

        BLOCK = min(4096, max(256, _next_pow2(N)))

        if N <= BLOCK:
            # Single-tile path — one program handles everything
            _hinge_loss_direct_kernel[(1,)](p, t, sum_buf, N,
                                            BLOCK=BLOCK)
        else:
            # Two-phase reduction path
            _hinge_loss_partial_kernel[(_NUM_PARTS,)](
                p, t, partial_buf, N,
                BLOCK=BLOCK, NUM_PARTS=_NUM_PARTS,
            )
            _hinge_loss_reduce_kernel[(1,)](
                partial_buf, sum_buf, N, NUM_PARTS=_NUM_PARTS,
            )

        return sum_buf[0]


batch_size = 32768
input_shape = (32768,)
dim = 1


def get_inputs():
    return [torch.rand(batch_size, *input_shape),
            torch.randint(0, 2, (batch_size,)).float() * 2 - 1]


def get_init_inputs():
    return []