import torch
import torch.nn as nn
import triton
import triton.language as tl

_MAX_PROGRAMS = 65535
_BLOCK = 1024
_FINAL_BLOCK = 1024


@triton.jit
def _kl_div_row_atomic_kernel(
    pred_ptr,
    targ_ptr,
    out_ptr,
    B,
    D,
    n_programs,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    tl.max_contiguous(offs, BLOCK_SIZE)
    tl.multiple_of(offs, BLOCK_SIZE)

    for row in range(pid, B, n_programs):
        base = row * D
        acc = tl.zeros((BLOCK_SIZE, ), dtype=tl.float32)
        n_iters = tl.cdiv(D, BLOCK_SIZE)
        for k in range(0, n_iters, 2):
            cols0 = k * BLOCK_SIZE + offs
            mask0 = cols0 < D
            idx0 = base + cols0
            p0 = tl.load(pred_ptr + idx0, mask=mask0, other=1.0).to(tl.float32)
            t0 = tl.load(targ_ptr + idx0, mask=mask0, other=0.0).to(tl.float32)
            t0_pos = t0 > 0.0
            t0_safe = tl.where(t0_pos, t0, 1.0)
            acc += tl.where(t0_pos, t0 * tl.log(t0_safe),
                            0.0) - t0 * tl.log(p0)

            k1 = k + 1
            if k1 < n_iters:
                cols1 = k1 * BLOCK_SIZE + offs
                mask1 = cols1 < D
                idx1 = base + cols1
                p1 = tl.load(pred_ptr + idx1, mask=mask1,
                             other=1.0).to(tl.float32)
                t1 = tl.load(targ_ptr + idx1, mask=mask1,
                             other=0.0).to(tl.float32)
                t1_pos = t1 > 0.0
                t1_safe = tl.where(t1_pos, t1, 1.0)
                acc += tl.where(t1_pos, t1 * tl.log(t1_safe),
                                0.0) - t1 * tl.log(p1)
        tl.atomic_add(out_ptr, tl.sum(acc, axis=0) / B, sem="relaxed")


@triton.jit
def _kl_div_row_contig_kernel(
    pred_ptr,
    targ_ptr,
    out_ptr,
    B,
    D,
    n_programs,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    tl.max_contiguous(offs, BLOCK_SIZE)
    tl.multiple_of(offs, BLOCK_SIZE)

    for row in range(pid, B, n_programs):
        base = row * D
        acc = tl.zeros((BLOCK_SIZE, ), dtype=tl.float32)
        n_iters = tl.cdiv(D, BLOCK_SIZE)
        for k in range(0, n_iters, 2):
            cols0 = k * BLOCK_SIZE + offs
            mask0 = cols0 < D
            idx0 = base + cols0
            p0 = tl.load(pred_ptr + idx0, mask=mask0, other=1.0).to(tl.float32)
            t0 = tl.load(targ_ptr + idx0, mask=mask0, other=0.0).to(tl.float32)
            t0_pos = t0 > 0.0
            t0_safe = tl.where(t0_pos, t0, 1.0)
            acc += tl.where(t0_pos, t0 * tl.log(t0_safe),
                            0.0) - t0 * tl.log(p0)

            k1 = k + 1
            if k1 < n_iters:
                cols1 = k1 * BLOCK_SIZE + offs
                mask1 = cols1 < D
                idx1 = base + cols1
                p1 = tl.load(pred_ptr + idx1, mask=mask1,
                             other=1.0).to(tl.float32)
                t1 = tl.load(targ_ptr + idx1, mask=mask1,
                             other=0.0).to(tl.float32)
                t1_pos = t1 > 0.0
                t1_safe = tl.where(t1_pos, t1, 1.0)
                acc += tl.where(t1_pos, t1 * tl.log(t1_safe),
                                0.0) - t1 * tl.log(p1)
        tl.store(out_ptr + row, tl.sum(acc, axis=0))


@triton.jit
def _kl_div_partial_kernel(
    pred_ptr,
    targ_ptr,
    partial_ptr,
    n_elements,
    n_tiles,
    n_programs,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    tl.max_contiguous(offs, BLOCK_SIZE)
    tl.multiple_of(offs, BLOCK_SIZE)
    acc = tl.zeros((BLOCK_SIZE, ), dtype=tl.float32)

    for tile in range(pid, n_tiles, n_programs):
        idx = tile * BLOCK_SIZE + offs
        mask = idx < n_elements
        p = tl.load(pred_ptr + idx, mask=mask, other=1.0).to(tl.float32)
        t = tl.load(targ_ptr + idx, mask=mask, other=0.0).to(tl.float32)
        t_pos = t > 0.0
        t_safe = tl.where(t_pos, t, 1.0)
        contrib = tl.where(t_pos, t * tl.log(t_safe), 0.0) - t * tl.log(p)
        acc += tl.where(mask, contrib, 0.0)

    tl.store(partial_ptr + pid, tl.sum(acc, axis=0))


@triton.jit
def _kl_div_finalize_kernel(
    partial_ptr,
    out_ptr,
    n_programs,
    batch_size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_programs
    vals = tl.load(partial_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    subtotal = tl.sum(vals, axis=0) / batch_size
    tl.atomic_add(out_ptr, subtotal, sem="relaxed")


class ModelNew(nn.Module):
    """KL divergence with batchmean reduction for probability inputs."""

    def __init__(self, use_triton_fallback: bool = False):
        super().__init__()
        self._use_triton_fallback = use_triton_fallback

    def forward(self, predictions, targets):
        if predictions.device.type != "npu" or targets.device.type != "npu":
            raise ValueError(
                "ModelNew expects predictions and targets on Ascend NPU")
        if predictions.ndim != 2 or targets.ndim != 2:
            raise ValueError("ModelNew expects 2D predictions and targets")
        if predictions.shape != targets.shape:
            raise ValueError(
                "ModelNew expects predictions and targets with matching shapes"
            )

        p = predictions if predictions.is_contiguous(
        ) else predictions.contiguous()
        t = targets if targets.is_contiguous() else targets.contiguous()
        B, D = p.shape

        if not self._use_triton_fallback:
            n_programs = min(B, _MAX_PROGRAMS)
            if B * D >= 50_000_000:
                row_sums = torch.empty((B, ),
                                       device=p.device,
                                       dtype=torch.float32)
                _kl_div_row_contig_kernel[(n_programs, )](
                    p,
                    t,
                    row_sums,
                    B,
                    D,
                    n_programs,
                    BLOCK_SIZE=_BLOCK,
                    num_warps=8,
                    num_stages=2,
                )
                return row_sums.sum() / B

            out = torch.empty((1, ), device=p.device, dtype=torch.float32)
            out.zero_()
            _kl_div_row_atomic_kernel[(n_programs, )](
                p,
                t,
                out,
                B,
                D,
                n_programs,
                BLOCK_SIZE=_BLOCK,
                num_warps=8,
                num_stages=2,
            )
            return out[0]

        n_elements = p.numel()
        n_tiles = triton.cdiv(n_elements, _BLOCK)
        n_programs = min(n_tiles, _MAX_PROGRAMS)
        partial = torch.empty((n_programs, ),
                              device=p.device,
                              dtype=torch.float32)
        out = torch.empty((1, ), device=p.device, dtype=torch.float32)
        out.zero_()
        _kl_div_partial_kernel[(n_programs, )](
            p,
            t,
            partial,
            n_elements,
            n_tiles,
            n_programs,
            BLOCK_SIZE=_BLOCK,
            num_warps=8,
            num_stages=2,
        )
        final_grid = (triton.cdiv(n_programs, _FINAL_BLOCK), )
        _kl_div_finalize_kernel[final_grid](
            partial,
            out,
            n_programs,
            B,
            BLOCK_SIZE=_FINAL_BLOCK,
            num_warps=8,
            num_stages=2,
        )
        return out[0]


batch_size = 8192 * 2
input_shape = (8192 * 2, )
dim = 1


def get_inputs():
    scale = torch.rand(())
    return [(torch.rand(batch_size, *input_shape) * scale).softmax(dim=-1),
            torch.rand(batch_size, *input_shape).softmax(dim=-1)]


def get_init_inputs():
    return []
