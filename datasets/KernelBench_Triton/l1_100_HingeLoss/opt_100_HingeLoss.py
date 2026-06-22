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

# ── Two-Phase Reduction Pattern ──────────────────────────────────
#
# Phase 1:  NUM_PARTS programs each compute a private partial sum
#           over its assigned tiles (no atomic operations).
# Phase 2:  Single program sums the partials and divides by N.
#
# This eliminates tl.atomic_add serialisation: every core writes to
# its own slot, then one core does a trivial vector sum.
#
# Adaptive sizing:
#   - BLOCK_SIZE = min(4096, max(256, next_pow2(N)))
#   - NUM_PARTS  = min(32, cdiv(N, BLOCK_SIZE))
#
# When n_tiles <= 32, each program covers exactly one contiguous tile
# (stripe degenerates to contiguous).  When n_tiles > 32, each of the
# 32 programs strides across the full range.

if _TRITON_AVAILABLE:

    @triton.jit
    def _hinge_loss_partial_kernel(
        pred_ptr,
        targ_ptr,
        partial_ptr,
        N,
        BLOCK: tl.constexpr,
        NUM_PARTS: tl.constexpr,
    ):
        """Phase 1 — each program accumulates a private partial sum."""
        pid = tl.program_id(axis=0)
        acc = 0.0
        n_chunks = tl.cdiv(N, NUM_PARTS * BLOCK)
        for chunk_idx in tl.range(0, n_chunks):
            chunk = pid * BLOCK + chunk_idx * NUM_PARTS * BLOCK
            offs = chunk + tl.arange(0, BLOCK)
            offs = tl.max_contiguous(tl.multiple_of(offs, BLOCK), BLOCK)
            mask = offs < N
            p = tl.load(pred_ptr + offs,
                        mask=mask,
                        other=1.0,
                        care_padding=False)
            t = tl.load(targ_ptr + offs,
                        mask=mask,
                        other=1.0,
                        care_padding=False)
            z = 1.0 - p * t
            z = tl.maximum(z, 0.0)
            acc += tl.sum(z, axis=0)
        tl.store(partial_ptr + pid, acc)

    @triton.jit
    def _hinge_loss_reduce_kernel(
        partial_ptr,
        out_ptr,
        N,
        NUM_PARTS: tl.constexpr,
    ):
        """Phase 2 — single program sums partials and divides by N."""
        vals = tl.load(partial_ptr + tl.arange(0, NUM_PARTS))
        tl.store(out_ptr, tl.sum(vals, axis=0) / N)

    @triton.jit
    def _hinge_loss_direct_kernel(
        pred_ptr,
        targ_ptr,
        out_ptr,
        N,
        BLOCK: tl.constexpr,
    ):
        """Single-tile kernel for N <= BLOCK_SIZE — no reduction needed."""
        offs = tl.arange(0, BLOCK)
        offs = tl.max_contiguous(tl.multiple_of(offs, BLOCK), BLOCK)
        mask = offs < N
        p = tl.load(pred_ptr + offs, mask=mask, other=1.0, care_padding=False)
        t = tl.load(targ_ptr + offs, mask=mask, other=1.0, care_padding=False)
        z = 1.0 - p * t
        z = tl.maximum(z, 0.0)
        part = tl.sum(z, axis=0)
        tl.store(out_ptr, part / N)


class ModelNew(nn.Module):
    """
    Hinge Loss for binary classification — optimised for Ascend NPU.

    Uses two-phase reduction (Phase 1: private partial sums, Phase 2:
    single-program reduce) to avoid tl.atomic_add serialisation.

    Adaptive block size and program count:
      - BLOCK_SIZE = min(4096, max(256, next_pow2(N)))
      - NUM_PARTS  = min(32, ceil(N / BLOCK_SIZE))

    Three-path dispatch:
      - N <= BLOCK_SIZE → direct single-tile kernel (1 program)
      - N  > BLOCK_SIZE → two-phase with NUM_PARTS partial programs
    """

    def __init__(self):
        super(ModelNew, self).__init__()

    @staticmethod
    def _next_pow2(x: int) -> int:
        return 1 if x <= 1 else 1 << (x - 1).bit_length()

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

        # ── Adaptive sizing (same as baseline block-selection logic) ──
        BLOCK_SIZE = min(4096, max(256, self._next_pow2(N)))
        n_tiles = triton.cdiv(N, BLOCK_SIZE)
        NUM_PARTS = min(32, n_tiles)

        # ── Three-path dispatch ────────────────────────────
        if N <= BLOCK_SIZE:
            # Direct single-tile: 1 program, no two-phase overhead
            out = torch.zeros(1, device=p.device, dtype=torch.float32)
            _hinge_loss_direct_kernel[(1, )](
                p,
                t,
                out,
                N,
                BLOCK=BLOCK_SIZE,
            )
            return out[0]

        # Two-phase reduction: private partials + single reduce
        partial_buf = torch.zeros(NUM_PARTS,
                                  device=p.device,
                                  dtype=torch.float32)
        out = torch.zeros(1, device=p.device, dtype=torch.float32)

        _hinge_loss_partial_kernel[(NUM_PARTS, )](
            p,
            t,
            partial_buf,
            N,
            BLOCK=BLOCK_SIZE,
            NUM_PARTS=NUM_PARTS,
        )
        _hinge_loss_reduce_kernel[(1, )](
            partial_buf,
            out,
            N,
            NUM_PARTS=NUM_PARTS,
        )
        return out[0]


# ── Test harness (used by profile_kernels.py) ─────────────────
batch_size = 32768
input_shape = (32768, )
dim = 1


def get_inputs():
    return [
        torch.rand(batch_size, *input_shape),
        torch.randint(0, 2, (batch_size, )).float() * 2 - 1,
    ]


def get_init_inputs():
    return []
