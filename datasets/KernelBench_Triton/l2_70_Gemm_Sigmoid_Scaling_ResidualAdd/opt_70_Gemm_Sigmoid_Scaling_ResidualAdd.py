import torch
import torch.nn as nn
import torch_npu  # noqa: F401
import triton
import triton.language as tl

DEFAULT_BATCH_SIZE = 1024
DEFAULT_INPUT_SIZE = 8192
DEFAULT_HIDDEN_SIZE = 8192
DEFAULT_SCALING_FACTOR = 2.0

_MAX_GRID = 65535
_BLOCK_SIZE = 16384


def _is_npu_tensor(x: torch.Tensor) -> bool:
    return bool(getattr(x, "is_npu", False))


@triton.jit
def _sigmoid_scale_residual_direct(x_ptr, out_ptr, n_elements, scale,
                                   BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0, care_padding=False)
    x_fp32 = x.to(tl.float32)
    y = x_fp32 + tl.sigmoid(x_fp32) * scale
    tl.store(out_ptr + offsets, y.to(x.dtype), mask=mask)


@triton.jit
def _sigmoid_scale_residual_persistent(x_ptr, out_ptr, n_elements, scale,
                                       n_programs, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    n_tiles = tl.cdiv(n_elements, BLOCK_SIZE)
    for tile_id in range(pid, n_tiles, n_programs):
        offsets = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0, care_padding=False)
        x_fp32 = x.to(tl.float32)
        y = x_fp32 + tl.sigmoid(x_fp32) * scale
        tl.store(out_ptr + offsets, y.to(x.dtype), mask=mask)


class ModelNew(nn.Module):
    """GEMM followed by y = gemm(x) + scale * sigmoid(gemm(x))."""

    def __init__(
        self,
        input_size: int = DEFAULT_INPUT_SIZE,
        hidden_size: int = DEFAULT_HIDDEN_SIZE,
        scaling_factor: float = DEFAULT_SCALING_FACTOR,
    ):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(input_size, hidden_size)
        self.scaling_factor = float(scaling_factor)

    def forward(self, x):
        if not _is_npu_tensor(x):
            raise RuntimeError("ModelNew expects input tensors on Ascend NPU")
        if x.requires_grad:
            raise RuntimeError(
                "ModelNew does not support autograd-enabled inputs")
        if x.dtype not in (torch.float16, torch.float32):
            raise RuntimeError(
                "ModelNew supports only float16 and float32 inputs")
        if not _is_npu_tensor(self.gemm.weight):
            raise RuntimeError(
                "ModelNew weights must be placed on Ascend NPU before execution"
            )
        if self.gemm.bias is not None and not _is_npu_tensor(self.gemm.bias):
            raise RuntimeError(
                "ModelNew bias must be placed on Ascend NPU before execution")

        x = self.gemm(x).contiguous()
        n_elements = x.numel()
        y = torch.empty_like(x)
        n_tiles = triton.cdiv(n_elements, _BLOCK_SIZE)
        if n_tiles > _MAX_GRID:
            n_programs = _MAX_GRID
            _sigmoid_scale_residual_persistent[(n_programs, )](
                x,
                y,
                n_elements,
                self.scaling_factor,
                n_programs,
                BLOCK_SIZE=_BLOCK_SIZE,
                num_warps=4,
                num_stages=2,
            )
        else:
            _sigmoid_scale_residual_direct[(n_tiles, )](
                x,
                y,
                n_elements,
                self.scaling_factor,
                BLOCK_SIZE=_BLOCK_SIZE,
                num_warps=4,
                num_stages=2,
            )
        return y


_MODEL_CACHE: dict[tuple[str, torch.dtype], ModelNew] = {}


def run_operator(x: torch.Tensor) -> torch.Tensor:
    key = (str(x.device), x.dtype)
    model = _MODEL_CACHE.get(key)
    if model is None:
        model = ModelNew().to(device=x.device, dtype=x.dtype)
        model.eval()
        _MODEL_CACHE[key] = model
    return model(x)


batch_size = 1024
input_size = 8192
hidden_size = 8192
scaling_factor = 2.0


def get_inputs():
    return [torch.rand(batch_size, input_size, device='npu')]


def get_init_inputs():
    return [input_size, hidden_size, scaling_factor]
