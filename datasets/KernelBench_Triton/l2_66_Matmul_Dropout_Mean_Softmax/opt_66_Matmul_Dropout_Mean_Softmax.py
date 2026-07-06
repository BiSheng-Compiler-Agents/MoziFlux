import torch
import torch.nn as nn
import triton
import triton.language as tl

DEFAULT_BATCH_SIZE = 128
DEFAULT_IN_FEATURES = 100
DEFAULT_OUT_FEATURES = 50
DEFAULT_DROPOUT_P = 0.2

_FILL_BLOCK_SMALL = 128
_FILL_BLOCK_LARGE = 1024
_MAX_PROGRAMS = 65535


@triton.jit
def _fill_ones_direct(out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    tl.multiple_of(offs, 16)
    tl.max_contiguous(offs, BLOCK_SIZE)
    ones = tl.full((BLOCK_SIZE,), 1.0, tl.float32)
    tl.store(out_ptr + offs, ones, mask=mask)


@triton.jit
def _fill_ones_persistent(out_ptr, n_elements, n_programs, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    n_tiles = tl.cdiv(n_elements, BLOCK_SIZE)
    for tile_id in range(pid, n_tiles, n_programs):
        offs = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements
        tl.multiple_of(offs, 16)
        tl.max_contiguous(offs, BLOCK_SIZE)
        ones = tl.full((BLOCK_SIZE,), 1.0, tl.float32)
        tl.store(out_ptr + offs, ones, mask=mask)


class ModelNew(nn.Module):
    """Analytic implementation of Linear -> Dropout -> mean(dim=1, keepdim=True) -> softmax.

    The softmax dimension has size 1, so the exact finite-output result is a tensor of ones
    with shape (batch_size, 1).  The constructor keeps the original signature but avoids
    materializing unused Linear/Dropout modules in the hot path.
    """

    def __init__(
        self,
        in_features=DEFAULT_IN_FEATURES,
        out_features=DEFAULT_OUT_FEATURES,
        dropout_p=DEFAULT_DROPOUT_P,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout_p = dropout_p

    def forward(self, x):
        if x.device.type != "npu":
            raise RuntimeError(
                "ModelNew expects Ascend NPU inputs and does not provide a PyTorch fallback path."
            )
        batch_size = x.shape[0]
        out = torch.empty((batch_size, 1), device=x.device, dtype=x.dtype)
        n_elements = out.numel()
        if n_elements == 0:
            return out

        block = _FILL_BLOCK_SMALL if n_elements <= _FILL_BLOCK_SMALL else _FILL_BLOCK_LARGE
        n_tiles = triton.cdiv(n_elements, block)
        if n_tiles > _MAX_PROGRAMS:
            _fill_ones_persistent[(_MAX_PROGRAMS,)](
                out, n_elements, _MAX_PROGRAMS, BLOCK_SIZE=block, num_warps=1, num_stages=2
            )
        else:
            _fill_ones_direct[(n_tiles,)](
                out, n_elements, BLOCK_SIZE=block, num_warps=1, num_stages=2
            )
        return out


batch_size = DEFAULT_BATCH_SIZE
in_features = DEFAULT_IN_FEATURES
out_features = DEFAULT_OUT_FEATURES
dropout_p = DEFAULT_DROPOUT_P


def get_inputs():
    return [torch.randn(batch_size, in_features, device="npu")]


def get_init_inputs():
    return [in_features, out_features, dropout_p]
