import torch
import torch.nn as nn
import torch_npu  # noqa: F401
import triton
import triton.language as tl

DEFAULT_BATCH_SIZE = 64
DEFAULT_IN_CHANNELS = 64
DEFAULT_OUT_CHANNELS = 64
DEFAULT_HEIGHT = 256
DEFAULT_WIDTH = 256
DEFAULT_KERNEL_SIZE = 3
DEFAULT_MULTIPLIER_SHAPE = (DEFAULT_OUT_CHANNELS, 1, 1)


def _is_npu_tensor(x: torch.Tensor) -> bool:
    return bool(getattr(x, "is_npu", False))


@triton.jit
def _fused_scale_lrelu_gelu(
    x_ptr,
    m_ptr,
    y_ptr,
    n_planes,
    C,
    HW,
    negative_slope: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    BLOCK_PLANES: tl.constexpr,
    CHUNKS_PER_PROGRAM: tl.constexpr,
):
    pid_hw = tl.program_id(axis=0)
    pid_plane = tl.program_id(axis=1) * BLOCK_PLANES + tl.arange(
        0, BLOCK_PLANES)
    plane_mask = pid_plane < n_planes
    scale = tl.load(m_ptr + (pid_plane % C), mask=plane_mask, other=1.0)
    s32 = scale[:, None].to(tl.float32)
    hw_base = pid_hw * BLOCK_HW * CHUNKS_PER_PROGRAM
    hw_range = tl.arange(0, BLOCK_HW)
    inv_sqrt2 = 0.7071067811865476

    for chunk_idx in tl.static_range(0, CHUNKS_PER_PROGRAM):
        hw_offsets = hw_base + chunk_idx * BLOCK_HW + hw_range
        hw_offsets = tl.max_contiguous(tl.multiple_of(hw_offsets, BLOCK_HW),
                                       BLOCK_HW)
        hw_mask = hw_offsets < HW
        mask = plane_mask[:, None] & hw_mask[None, :]
        base = pid_plane[:, None] * HW + hw_offsets[None, :]
        x = tl.load(x_ptr + base, mask=mask, other=0.0, cache_modifier=".cg")
        x32 = x.to(tl.float32)
        v = x32 * s32
        v = v + (negative_slope - 1.0) * tl.minimum(v, 0.0)
        e = tl.erf(v * inv_sqrt2)
        y32 = 0.5 * v * (1.0 + e)
        tl.store(y_ptr + base, y32.to(x.dtype), mask=mask)


class ModelNew(nn.Module):

    def __init__(
        self,
        in_channels=DEFAULT_IN_CHANNELS,
        out_channels=DEFAULT_OUT_CHANNELS,
        kernel_size=DEFAULT_KERNEL_SIZE,
        multiplier_shape=DEFAULT_MULTIPLIER_SHAPE,
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x):
        if not _is_npu_tensor(x):
            raise RuntimeError("ModelNew expects input tensors on Ascend NPU")
        if x.requires_grad:
            raise RuntimeError(
                "ModelNew does not support autograd-enabled inputs")

        x = self.conv(x).contiguous()
        n, c, h, w = x.shape
        if self.multiplier.numel() != c:
            raise RuntimeError(
                f"Multiplier must contain exactly one value per channel, got {self.multiplier.numel()} for C={c}"
            )

        out = torch.empty_like(x)
        m = self.multiplier.contiguous().reshape(-1).to(device=x.device,
                                                        dtype=x.dtype)
        hw = h * w
        n_planes = n * c
        if hw >= 2048:
            block_hw = 2048
            block_planes = 4
            chunks_per_program = 4
            num_warps = 8
            num_stages = 2
        elif hw >= 1024:
            block_hw = 1024
            block_planes = 4
            chunks_per_program = 4
            num_warps = 4
            num_stages = 2
        else:
            block_hw = 512
            block_planes = 4
            chunks_per_program = 1
            num_warps = 4
            num_stages = 2

        grid_hw = triton.cdiv(hw, block_hw * chunks_per_program)
        grid_plane = triton.cdiv(n_planes, block_planes)
        while grid_hw * grid_plane >= 65536:
            if block_planes < 16:
                block_planes *= 2
                grid_plane = triton.cdiv(n_planes, block_planes)
                continue
            if chunks_per_program < 4:
                chunks_per_program *= 2
            elif block_hw < 2048:
                block_hw *= 2
            else:
                raise RuntimeError(
                    "Unable to satisfy Triton grid<65536 bound with the current plane-major mapping"
                )
            grid_hw = triton.cdiv(hw, block_hw * chunks_per_program)

        def grid(meta):
            return (
                triton.cdiv(hw, meta["BLOCK_HW"] * meta["CHUNKS_PER_PROGRAM"]),
                triton.cdiv(n_planes, meta["BLOCK_PLANES"]),
            )

        _fused_scale_lrelu_gelu[grid](
            x,
            m,
            out,
            n_planes,
            c,
            hw,
            self.leaky_relu.negative_slope,
            BLOCK_HW=block_hw,
            BLOCK_PLANES=block_planes,
            CHUNKS_PER_PROGRAM=chunks_per_program,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        return out


_MODEL_CACHE: dict[tuple[torch.device, torch.dtype], ModelNew] = {}


def run_operator(x: torch.Tensor) -> torch.Tensor:
    key = (x.device, x.dtype)
    model = _MODEL_CACHE.get(key)
    if model is None:
        model = ModelNew().to(device=x.device, dtype=x.dtype)
        model.eval()
        _MODEL_CACHE[key] = model
    return model(x)


batch_size = 64
in_channels = 64
out_channels = 64
height, width = 256, 256
kernel_size = 3
multiplier_shape = (out_channels, 1, 1)


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width, device="npu")]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, multiplier_shape]
