import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


def _is_npu_tensor(x: torch.Tensor) -> bool:
    return bool(getattr(x, 'is_npu', False) or x.device.type == 'npu')


@triton.jit
def _generic_reduce_bcdhw_to_b_kernel(
    x_ptr,
    out_ptr,
    stride_b,
    stride_c,
    C,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < C
    vals = tl.load(x_ptr + pid * stride_b + offs * stride_c, mask=mask, other=0.0)
    total = tl.sum(vals.to(tl.float32), axis=0)
    tl.store(out_ptr + pid, total)


@triton.jit
def _reduce_bcdhw_to_b_kernel(
    x_ptr,
    out_ptr,
    stride_b,
    stride_c,
    B,
    BLOCK_M: tl.constexpr,
    GROUP_TILES: tl.constexpr,
):
    pid = tl.program_id(0)
    cols = tl.arange(0, 16)
    row_step = stride_b
    col_step = stride_c
    base_row = pid * 64
    for tile_idx in tl.static_range(0, GROUP_TILES):
        rows = base_row + tile_idx * BLOCK_M + tl.arange(0, BLOCK_M)
        row_mask = rows < B
        ptrs = x_ptr + rows[:, None] * row_step + cols[None, :] * col_step
        vals = tl.load(ptrs, mask=row_mask[:, None], other=0.0)
        vals_f32 = vals.to(tl.float32)
        acc = tl.sum(vals_f32, axis=1)
        tl.store(out_ptr + rows, acc.to(vals.dtype), mask=row_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, divisor, pool_size, bias_shape, sum_dim):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.divisor = divisor
        self.max_pool = nn.MaxPool3d(pool_size)
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.sum_dim = sum_dim

    def forward(self, x):
        if not _is_npu_tensor(x):
            raise RuntimeError(f'ModelNew expects NPU input, got {x.device.type}')
        if self.sum_dim != 1:
            raise RuntimeError(f'ModelNew only supports sum_dim == 1 for the Triton path, got {self.sum_dim}')
        w = self.conv.weight
        b = self.conv.bias
        x = F.conv3d(
            x,
            w / self.divisor,
            None if b is None else b / self.divisor,
            stride=self.conv.stride,
            padding=self.conv.padding,
            dilation=self.conv.dilation,
            groups=self.conv.groups,
        )
        x = self.max_pool(x)
        x = self.global_avg_pool(x)
        x = (x + self.bias).reshape(x.shape[0], x.shape[1]).contiguous()
        B, C = x.shape
        out = torch.empty((B,), device=x.device, dtype=x.dtype)
        if C == 16:
            grid = (triton.cdiv(B, 64),)
            _reduce_bcdhw_to_b_kernel[grid](
                x,
                out,
                x.stride(0),
                x.stride(1),
                B,
                BLOCK_M=64,
                GROUP_TILES=1,
                num_warps=4,
                num_stages=2,
            )
        else:
            if C > 256:
                raise RuntimeError(f'ModelNew only supports up to 256 output channels, got {C}')
            _generic_reduce_bcdhw_to_b_kernel[(B,)](
                x,
                out,
                x.stride(0),
                x.stride(1),
                C,
                BLOCK_SIZE=256,
                num_warps=4,
                num_stages=2,
            )
        return out.view(B, 1, 1, 1)


_MODEL_CACHE: dict[tuple[int | None, torch.dtype], ModelNew] = {}


def conv3d_divide_max_globalavgpool_biasadd_sum(x):
    if not _is_npu_tensor(x):
        raise RuntimeError(
            f'conv3d_divide_max_globalavgpool_biasadd_sum expects NPU input, got {x.device.type}'
        )
    key = (getattr(x.device, 'index', None), torch.float32)
    model = _MODEL_CACHE.get(key)
    if model is None:
        torch.manual_seed(0)
        model = ModelNew(*get_init_inputs()).to(device=x.device, dtype=torch.float32).eval()
        _MODEL_CACHE[key] = model
    with torch.no_grad():
        return model(x.to(dtype=torch.float32))


batch_size = 128
in_channels = 8
out_channels = 16
depth = 16
height = 64
width = 64
kernel_size = (3, 3, 3)
divisor = 2.0
pool_size = (2, 2, 2)
bias_shape = (out_channels, 1, 1, 1)
sum_dim = 1


def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, divisor, pool_size, bias_shape, sum_dim]
