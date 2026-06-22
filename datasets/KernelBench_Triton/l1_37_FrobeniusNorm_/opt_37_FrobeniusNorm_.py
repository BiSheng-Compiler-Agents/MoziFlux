import torch
import torch.nn as nn
import torch_npu  # noqa: F401
import triton
import triton.language as tl

_MAX_PROGRAMS = 65535
_BLOCK_ELEMS = 8192
_REDUCE_BLOCK = 1024


@triton.jit
def _sumsq_atomic_kernel(x_ptr, n_elements, out_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    mask = idx < n_elements
    x = tl.load(x_ptr + idx, mask=mask, other=0.0,
                care_padding=False).to(tl.float32)
    ss = tl.sum(x * x, axis=0)
    tl.atomic_add(out_ptr, ss)


@triton.jit
def _partial_sumsq_kernel(x_ptr, partials_ptr, n_elements, n_tiles, n_programs,
                          BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    tile = pid
    offs = tl.arange(0, BLOCK)
    while tile < n_tiles:
        idx = tile * BLOCK + offs
        mask = idx < n_elements
        x = tl.load(x_ptr + idx, mask=mask, other=0.0,
                    care_padding=False).to(tl.float32)
        ss = tl.sum(x * x, axis=0)
        tl.store(partials_ptr + tile, ss)
        tile += n_programs


@triton.jit
def _reduce_partials_kernel(partials_ptr, sumsq_ptr, n_partials,
                            BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    acc = tl.zeros((), dtype=tl.float32)
    start = 0
    while start < n_partials:
        idx = start + offs
        mask = idx < n_partials
        vals = tl.load(partials_ptr + idx,
                       mask=mask,
                       other=0.0,
                       care_padding=False)
        acc += tl.sum(vals, axis=0)
        start += BLOCK
    tl.store(sumsq_ptr, acc)


@triton.jit
def _scale_kernel(x_ptr, y_ptr, n_elements, sumsq_ptr, n_tiles, n_programs,
                  BLOCK: tl.constexpr):
    ss = tl.load(sumsq_ptr)
    inv_norm = tl.rsqrt(ss)
    pid = tl.program_id(0)
    tile = pid
    offs = tl.arange(0, BLOCK)
    while tile < n_tiles:
        idx = tile * BLOCK + offs
        mask = idx < n_elements
        x = tl.load(x_ptr + idx, mask=mask, other=0.0,
                    care_padding=False).to(tl.float32)
        y = x * inv_norm
        tl.store(y_ptr + idx, y, mask=mask)
        tile += n_programs


class ModelNew(nn.Module):
    """Frobenius norm normalization: y = x / sqrt(sum(x*x))."""

    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.device.type != "npu":
            raise ValueError("ModelNew expects an Ascend NPU tensor")
        if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError(
                f"ModelNew supports float16, bfloat16, and float32 inputs, got {x.dtype}"
            )
        if x.numel() == 0:
            raise ValueError("ModelNew does not support empty tensors")

        original_shape = x.shape
        x_contig = x if x.is_contiguous() else x.contiguous()
        n_elements = x_contig.numel()
        n_tiles = triton.cdiv(n_elements, _BLOCK_ELEMS)
        n_programs = min(n_tiles, _MAX_PROGRAMS)
        if n_programs < 1:
            n_programs = 1

        y = torch.empty_like(x_contig, dtype=torch.float32)

        if n_tiles <= _MAX_PROGRAMS:
            # Fast direct path: keep the original two-launch algorithm when the grid is legal.
            sumsq = torch.zeros((1, ),
                                device=x_contig.device,
                                dtype=torch.float32)
            _sumsq_atomic_kernel[(n_tiles, )](x_contig,
                                              n_elements,
                                              sumsq,
                                              BLOCK=_BLOCK_ELEMS)
        else:
            # Overflow-safe path: avoid Ascend coreDim > 65535 and avoid global atomic serialization.
            partials = torch.empty((n_tiles, ),
                                   device=x_contig.device,
                                   dtype=torch.float32)
            sumsq = torch.empty((1, ),
                                device=x_contig.device,
                                dtype=torch.float32)
            _partial_sumsq_kernel[(n_programs, )](x_contig,
                                                  partials,
                                                  n_elements,
                                                  n_tiles,
                                                  n_programs,
                                                  BLOCK=_BLOCK_ELEMS)
            _reduce_partials_kernel[(1, )](partials,
                                           sumsq,
                                           n_tiles,
                                           BLOCK=_REDUCE_BLOCK)

        _scale_kernel[(n_programs, )](x_contig,
                                      y,
                                      n_elements,
                                      sumsq,
                                      n_tiles,
                                      n_programs,
                                      BLOCK=_BLOCK_ELEMS)
        return y.view(original_shape)


batch_size = 112
features = 64
dim1 = 512
dim2 = 512


def get_inputs():
    x = torch.rand(batch_size, features, dim1, dim2)
    return [x]


def get_init_inputs():
    return []
