import torch
import torch.nn as nn
import torch_npu  # noqa: F401
import triton
import triton.language as tl

DEFAULT_BATCH_SIZE = 1024
DEFAULT_INPUT_SIZE = 8192
DEFAULT_HIDDEN_SIZE = 8192
DEFAULT_SCALING_FACTOR = 2.0


def _is_npu_tensor(x: torch.Tensor) -> bool:
    return bool(getattr(x, "is_npu", False))


@triton.jit
def _sigmoid_scale_residual_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    scale,
    CAST_TO_FP32: tl.constexpr,
    USE_ALIGNMENT_HINTS: tl.constexpr,
    USE_STREAMING_LOAD: tl.constexpr,
    USE_EVICT_STORE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    if USE_ALIGNMENT_HINTS:
        offsets = tl.max_contiguous(tl.multiple_of(offsets, BLOCK_SIZE),
                                    BLOCK_SIZE)

    if USE_STREAMING_LOAD:
        x = tl.load(x_ptr + offsets,
                    mask=mask,
                    other=0.0,
                    cache_modifier=".cg")
    else:
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    if CAST_TO_FP32:
        x_math = x.to(tl.float32)
    else:
        x_math = x

    y = tl.fma(tl.sigmoid(x_math), scale, x_math)

    if CAST_TO_FP32:
        y_store = y.to(x.dtype)
    else:
        y_store = y

    if USE_EVICT_STORE:
        tl.store(out_ptr + offsets,
                 y_store,
                 mask=mask,
                 eviction_policy="evict_first")
    else:
        tl.store(out_ptr + offsets, y_store, mask=mask)


def _resolve_launch_config(
        n_elements: int,
        dtype: torch.dtype) -> tuple[int, int, int, bool, bool, bool, bool]:
    if dtype == torch.float16:
        if n_elements >= (1 << 18):
            return 4096, 4, 1, True, True, False, False
        return 2048, 4, 1, True, True, False, False

    if n_elements >= (1 << 22):
        return 16384, 8, 2, False, True, False, False
    if n_elements >= (1 << 18):
        return 8192, 4, 1, False, True, False, False
    return 4096, 4, 1, False, True, False, False


class ModelNew(nn.Module):
    """
    Model implementing the pattern "Gemm_Sigmoid_Scaling_ResidualAdd".
    """

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
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, hidden_size).
        """
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

        x = self.gemm(x)
        x = x.contiguous()
        n_elements = x.numel()
        y = torch.empty_like(x)

        (
            block_size,
            num_warps,
            num_stages,
            cast_to_fp32,
            use_alignment_hints,
            use_streaming_load,
            use_evict_store,
        ) = _resolve_launch_config(n_elements, x.dtype)

        def grid(META):
            return (triton.cdiv(n_elements, META["BLOCK_SIZE"]), )

        _sigmoid_scale_residual_kernel[grid](
            x,
            y,
            n_elements,
            self.scaling_factor,
            CAST_TO_FP32=cast_to_fp32,
            USE_ALIGNMENT_HINTS=use_alignment_hints,
            USE_STREAMING_LOAD=use_streaming_load,
            USE_EVICT_STORE=use_evict_store,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
            num_stages=num_stages,
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
