import torch
import torch.nn as nn
import torch_npu  # noqa: F401
import triton
import triton.language as tl


DEFAULT_BATCH_SIZE = 1024
DEFAULT_IN_FEATURES = 8192
DEFAULT_OUT_FEATURES = 8192
DEFAULT_BN_EPS = 1e-05
DEFAULT_BN_MOMENTUM = 0.1
DEFAULT_BIAS_SHAPE = (1,)
DEFAULT_DIVIDE_VALUE = 1.0


def _is_npu_tensor(x: torch.Tensor) -> bool:
    return bool(getattr(x, "is_npu", False) or x.device.type == "npu")


@triton.jit
def _fused_bias_div_swish_flat_kernel(
    x_ptr,        # *float32, flattened [M*N]
    y_ptr,        # *float32, flattened [M*N] (can alias x_ptr for in-place)
    bias_ptr,     # *float32, shape (1,) scalar bias
    inv_div,      # float32 scalar = 1.0 / divide_value
    N_ELEMENTS,   # total number of elements = M * N
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N_ELEMENTS

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Load scalar bias once per program
    x_fp32 = x.to(tl.float32)
    b_fp32 = tl.load(bias_ptr).to(tl.float32)

    # z = (x + b) * inv_div
    z = (x_fp32 + b_fp32) * inv_div

    # Swish: z * sigmoid(z)
    s = 1.0 / (1.0 + tl.exp(-z))
    y = (z * s).to(x.dtype)

    tl.store(y_ptr + offsets, y, mask=mask)


class ModelNew(nn.Module):
    """
    Model that performs a matrix multiplication, batch normalization, bias addition, division, and Swish activation.
    """
    def __init__(
        self,
        in_features=DEFAULT_IN_FEATURES,
        out_features=DEFAULT_OUT_FEATURES,
        bn_eps=DEFAULT_BN_EPS,
        bn_momentum=DEFAULT_BN_MOMENTUM,
        bias_shape=DEFAULT_BIAS_SHAPE,
        divide_value=DEFAULT_DIVIDE_VALUE,
    ):
        super(ModelNew, self).__init__()
        if tuple(bias_shape) != (1,):
            raise ValueError("ModelNew supports only scalar bias_shape=(1,) for the fused Triton kernel")
        if float(divide_value) == 0.0:
            raise ValueError("divide_value must be non-zero")
        self.matmul = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features, eps=bn_eps, momentum=bn_momentum)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.divide_value = float(divide_value)

    def forward(self, x):
        if not _is_npu_tensor(x):
            raise RuntimeError("ModelNew expects an Ascend NPU tensor input")
        if x.requires_grad:
            raise RuntimeError("ModelNew does not support autograd-enabled inputs")
        x = self.matmul(x)
        x = self.bn(x)

        bias_dev = self.bias.to(device=x.device, dtype=x.dtype)
        n_elems = x.numel()
        if n_elems == 0:
            return x

        block_size = 1024
        grid = (triton.cdiv(n_elems, block_size),)
        inv_div = 1.0 / self.divide_value

        _fused_bias_div_swish_flat_kernel[grid](
            x,
            x,
            bias_dev,
            inv_div,
            n_elems,
            BLOCK_SIZE=block_size,
            num_warps=8,
            num_stages=3,
        )
        return x
batch_size = 1024
in_features = 8192
out_features = 8192
bn_eps = 1e-5
bn_momentum = 0.1
bias_shape = (1,)
divide_value = 1.0

def get_inputs():
    return [torch.rand(batch_size, in_features)]
def get_init_inputs():
    return [in_features, out_features, bn_eps, bn_momentum, bias_shape, divide_value]