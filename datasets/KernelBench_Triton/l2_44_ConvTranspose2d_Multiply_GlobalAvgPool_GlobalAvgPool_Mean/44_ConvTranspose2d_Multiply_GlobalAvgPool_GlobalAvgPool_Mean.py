import torch
import torch.nn as nn
import triton
import triton.language as tl

_MODEL_CACHE = {}


@triton.jit
def _global_avg_mul_kernel(
    x_ptr,
    out_ptr,
    NC,
    HW,
    multiplier,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    in_bounds = pid < NC
    base = x_ptr + pid * HW
    offs = tl.arange(0, BLOCK_SIZE)

    acc = tl.zeros((), dtype=tl.float32)
    start = 0
    while start < HW:
        idx0 = start + offs
        mask0 = (idx0 < HW) & in_bounds
        vals0 = tl.load(base + idx0, mask=mask0, other=0.0)
        acc += tl.sum(vals0.to(tl.float32), axis=0)

        idx1 = idx0 + BLOCK_SIZE
        mask1 = (idx1 < HW) & in_bounds
        vals1 = tl.load(base + idx1, mask=mask1, other=0.0)
        acc += tl.sum(vals1.to(tl.float32), axis=0)
        start += 2 * BLOCK_SIZE

    tl.store(out_ptr + pid, acc * multiplier / HW, mask=in_bounds)


def _fused_mul_global_avg(x: torch.Tensor, multiplier: float) -> torch.Tensor:
    if x.device.type != "npu":
        raise RuntimeError("_fused_mul_global_avg expects an Ascend NPU tensor")
    if x.ndim != 4:
        raise RuntimeError("_fused_mul_global_avg expects an NCHW tensor")

    x = x.contiguous()
    N, C, H, W = x.shape
    HW = H * W
    NC = N * C

    out = torch.empty((N, C, 1, 1), device=x.device, dtype=x.dtype)

    def next_pow2(v: int) -> int:
        return 1 << (v - 1).bit_length()

    bs = min(max(128, next_pow2(HW)), 2048)
    num_warps = 4 if bs <= 1024 else 8
    num_stages = 4

    grid = (NC,)
    _global_avg_mul_kernel[grid](
        x.view(-1),
        out.view(-1),
        NC,
        HW,
        float(multiplier),
        BLOCK_SIZE=bs,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out


class ModelNew(nn.Module):
    """
    Model that performs a transposed convolution, multiplies by a scalar, applies global average pooling, 
    another global average pooling
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, multiplier):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.multiplier = multiplier

    def forward(self, x):
        if x.device.type != "npu":
            raise RuntimeError("ModelNew expects inputs on Ascend NPU")
        if self.conv_transpose.weight.device.type != "npu":
            raise RuntimeError("ModelNew expects ConvTranspose2d weights on Ascend NPU")
        x = self.conv_transpose(x)
        x = _fused_mul_global_avg(x, self.multiplier)
        return x


def _build_model_for(device: torch.device, dtype: torch.dtype) -> ModelNew:
    with torch.random.fork_rng():
        torch.manual_seed(0)
        model = ModelNew(*get_init_inputs())
    model = model.to(device=device, dtype=dtype)
    model.eval()
    return model


def run_operator(x: torch.Tensor) -> torch.Tensor:
    if x.device.type != "npu":
        raise RuntimeError("run_operator expects inputs on Ascend NPU")

    key = (str(x.device), str(x.dtype))
    model = _MODEL_CACHE.get(key)
    if model is None:
        model = _build_model_for(x.device, x.dtype)
        _MODEL_CACHE[key] = model

    with torch.no_grad():
        return model(x)
batch_size = 16
in_channels = 64
out_channels = 128
height, width = 128, 128
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
multiplier = 0.5

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width, device='npu')]
def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, multiplier]