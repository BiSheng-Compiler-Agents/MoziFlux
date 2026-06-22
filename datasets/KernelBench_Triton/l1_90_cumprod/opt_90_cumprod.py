import torch
import torch.nn as nn
import triton
import triton.language as tl

_MAX_PROGRAMS = 65535
_BLOCK = 512


@triton.jit
def _cumprod_rowwise_block_kernel(
    x_ptr,
    y_ptr,
    M,
    N,
    stride_xm,
    stride_xn,
    stride_ym,
    stride_yn,
    BLOCK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    rel = tl.arange(0, BLOCK)
    x_row_ptr = x_ptr + pid_m * stride_xm
    y_row_ptr = y_ptr + pid_m * stride_ym
    carry = tl.full((), 1.0, dtype=tl.float32)

    for start in tl.range(0, N, BLOCK):
        offs = start + rel
        mask = offs < N
        vals = tl.load(x_row_ptr + offs * stride_xn, mask=mask,
                       other=1.0).to(tl.float32)
        scan = tl.cumprod(vals, axis=0)
        out = scan * carry
        tl.store(y_row_ptr + offs * stride_yn, out, mask=mask)
        valid = tl.minimum(BLOCK, N - start)
        last_rel = valid - 1
        block_last = tl.sum(tl.where(rel == last_rel, scan, 0.0), axis=0)
        carry *= block_last


@triton.jit
def _cumprod_rowwise_block_persistent_kernel(
    x_ptr,
    y_ptr,
    M,
    N,
    stride_xm,
    stride_xn,
    stride_ym,
    stride_yn,
    n_programs,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    rel = tl.arange(0, BLOCK)
    while row < M:
        x_row_ptr = x_ptr + row * stride_xm
        y_row_ptr = y_ptr + row * stride_ym
        carry = tl.full((), 1.0, dtype=tl.float32)
        for start in tl.range(0, N, BLOCK):
            offs = start + rel
            mask = offs < N
            vals = tl.load(x_row_ptr + offs * stride_xn, mask=mask,
                           other=1.0).to(tl.float32)
            scan = tl.cumprod(vals, axis=0)
            out = scan * carry
            tl.store(y_row_ptr + offs * stride_yn, out, mask=mask)
            valid = tl.minimum(BLOCK, N - start)
            last_rel = valid - 1
            block_last = tl.sum(tl.where(rel == last_rel, scan, 0.0), axis=0)
            carry *= block_last
        row += n_programs


class ModelNew(nn.Module):
    """Cumulative product along ``dim`` using a block-prefix Triton row scan."""

    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        if not x.is_npu:
            raise RuntimeError("ModelNew expects inputs on Ascend NPU")
        if x.ndim == 0:
            raise RuntimeError("cumprod requires at least one dimension")

        dim = self.dim
        if dim < 0:
            dim += x.ndim
        if dim < 0 or dim >= x.ndim:
            raise IndexError(
                f"dim={self.dim} is out of range for ndim={x.ndim}")
        if x.numel() == 0:
            return torch.empty_like(x)

        perm = [idx for idx in range(x.ndim) if idx != dim] + [dim]
        moved = x.permute(perm).contiguous()
        scan_len = moved.shape[-1]
        flat = moved.reshape(-1, scan_len)
        out_flat = torch.empty_like(flat)
        rows = flat.shape[0]

        if rows <= _MAX_PROGRAMS:
            _cumprod_rowwise_block_kernel[(rows, )](
                flat,
                out_flat,
                rows,
                scan_len,
                flat.stride(0),
                flat.stride(1),
                out_flat.stride(0),
                out_flat.stride(1),
                BLOCK=_BLOCK,
            )
        else:
            n_programs = _MAX_PROGRAMS
            _cumprod_rowwise_block_persistent_kernel[(n_programs, )](
                flat,
                out_flat,
                rows,
                scan_len,
                flat.stride(0),
                flat.stride(1),
                out_flat.stride(0),
                out_flat.stride(1),
                n_programs,
                BLOCK=_BLOCK,
            )

        out = out_flat.reshape(moved.shape)
        inv_perm = [0] * x.ndim
        for idx, src in enumerate(perm):
            inv_perm[src] = idx
        return out.permute(inv_perm)
