import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except Exception:
    _TRITON_AVAILABLE = False

if _TRITON_AVAILABLE:
    @triton.jit
    def _fused_sub_tanh_sub_kernel(
        x_ptr,
        y_ptr,
        n_elements,
        subtract1,
        subtract2,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements

        # Load
        x = tl.load(x_ptr + offs, mask=mask, other=0.0)

        # x - subtract1
        x = x - subtract1

        # Numerically-stable tanh using exponentials
        ax = tl.abs(x)
        e = tl.exp(-2.0 * ax)
        tanh_pos = (1.0 - e) / (1.0 + e)
        sign = tl.where(x >= 0.0, 1.0, -1.0)
        x = sign * tanh_pos

        # - subtract2 and store
        x = x - subtract2
        tl.store(y_ptr + offs, x, mask=mask)

    @triton.jit
    def _fused_tanh_avgpool_sub2_kernel(
        x_ptr,  # float* [N, C, H, W]
        y_ptr,  # float* [N, C, outH, outW]
        N, C, H, W,  # input dims
        outH, outW,  # output dims (after pooling)
        subtract1,   # float
        subtract2,   # float
        BLOCK_HW: tl.constexpr,  # number of output HW elements per program
        K: tl.constexpr,         # pooling kernel size; stride assumed = K (AvgPool2d default)
    ):
        # program ids
        pid_nc = tl.program_id(axis=0)  # over N*C
        pid_blk = tl.program_id(axis=1)  # over blocks of outH*outW

        # decode n, c
        n = pid_nc // C
        c = pid_nc % C

        # per-NC base strides
        in_nc_stride = H * W
        out_nc_stride = outH * outW

        # output linear indices this program handles
        offs_hw = pid_blk * BLOCK_HW + tl.arange(0, BLOCK_HW)
        mask_hw = offs_hw < (outH * outW)

        # map to (oh, ow)
        oh = offs_hw // outW
        ow = offs_hw % outW

        # starting input coordinates for each output (stride = K)
        ih0 = oh * K
        iw0 = ow * K

        # base pointers per (oh, ow) for the current (n, c)
        base_in = (n * C + c) * in_nc_stride + ih0 * W + iw0
        base_out = (n * C + c) * out_nc_stride + oh * outW + ow

        # accumulate tanh(x - subtract1) over KxK window
        acc = tl.zeros([BLOCK_HW], dtype=tl.float32)

        # Unrolled pooling window
        for rr in range(0, K):
            row_off = base_in + rr * W
            for ss in range(0, K):
                idx = row_off + ss
                x = tl.load(x_ptr + idx, mask=mask_hw, other=0.0)
                # stable tanh: tanh(x) = sign(x)*(1 - e)/(1 + e) where e=exp(-2*|x|)
                x = x - subtract1
                ax = tl.abs(x)
                e = tl.exp(-2.0 * ax)
                t = (1.0 - e) / (1.0 + e)
                x = tl.where(x >= 0.0, t, -t)
                acc += x

        inv_area = 1.0 / (K * K)
        out_val = acc * inv_area - subtract2
        tl.store(y_ptr + base_out, out_val, mask=mask_hw)


class ModelNew(nn.Module):
    """
    Model that performs a convolution, subtraction, tanh activation, subtraction and average pooling.
    """
    def __init__(self, in_channels, out_channels, kernel_size, subtract1_value, subtract2_value, kernel_size_pool):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract1_value = subtract1_value
        self.subtract2_value = subtract2_value
        self.avgpool = nn.AvgPool2d(kernel_size_pool)
        self.kernel_size_pool = kernel_size_pool

    def forward(self, x):
        x = self.conv(x)
        if not _TRITON_AVAILABLE:
            raise RuntimeError("Triton is required for ModelNew forward")
        if x.device.type != "npu":
            raise RuntimeError(f"ModelNew expects NPU inputs, got {x.device.type}")
        if x.dtype != torch.float32:
            raise RuntimeError(f"ModelNew expects float32 inputs, got {x.dtype}")

        x = x.contiguous()
        N, C, H, W = x.shape
        K = int(self.kernel_size_pool)
        outH = (H - K) // K + 1
        outW = (W - K) // K + 1
        y = torch.empty((N, C, outH, outW), device=x.device, dtype=x.dtype)

        BLOCK_HW = 1024
        grid = lambda meta: (N * C, triton.cdiv(outH * outW, meta["BLOCK_HW"]))
        _fused_tanh_avgpool_sub2_kernel[grid](
            x,
            y,
            N,
            C,
            H,
            W,
            outH,
            outW,
            float(self.subtract1_value),
            float(self.subtract2_value),
            BLOCK_HW=BLOCK_HW,
            K=K,
            num_warps=4,
            num_stages=2,
        )
        return y


_MODEL_CACHE = {}


def conv2d_subtract_tanh_subtract_avgpool(x):
    if x.device.type != "npu":
        raise RuntimeError(f"conv2d_subtract_tanh_subtract_avgpool expects NPU input, got {x.device.type}")

    cache_key = (x.device.type, x.dtype)
    module = _MODEL_CACHE.get(cache_key)
    if module is None:
        torch.manual_seed(0)
        module = ModelNew(*get_init_inputs()).to(device=x.device, dtype=torch.float32).eval()
        _MODEL_CACHE[cache_key] = module

    with torch.no_grad():
        return module(x.to(dtype=torch.float32))
batch_size = 128
in_channels = 64
out_channels = 128
height, width = 128, 128
kernel_size = 3
subtract1_value = 0.5
subtract2_value = 0.2
kernel_size_pool = 2

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]
def get_init_inputs():
    return [in_channels, out_channels, kernel_size, subtract1_value, subtract2_value, kernel_size_pool]