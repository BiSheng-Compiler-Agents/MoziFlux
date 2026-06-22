import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import triton.runtime.driver as driver

DEFAULT_KERNEL_SIZE = 8
DEFAULT_STRIDE = 1
DEFAULT_PADDING = 4
DEFAULT_DILATION = 3
DEFAULT_RETURN_INDICES = False
_MAX_PROGRAMS = 65535


@triton.jit
def _maxpool1d_noindex_direct_kernel(
    x_ptr,
    y_ptr,
    L_in,
    L_out,
    STRIDE,
    PADDING,
    DILATION,
    line_stride_x,
    line_stride_y,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid_nc = tl.program_id(axis=0)
    pid_o_blk = tl.program_id(axis=1)

    o_offsets = pid_o_blk * BLOCK + tl.arange(0, BLOCK)
    mask_o = o_offsets < L_out
    starts = o_offsets * STRIDE - PADDING

    base_x = x_ptr + pid_nc.to(tl.int64) * line_stride_x
    base_y = y_ptr + pid_nc.to(tl.int64) * line_stride_y

    tl.multiple_of(o_offsets, 16)
    tl.max_contiguous(o_offsets, BLOCK)

    block_start = pid_o_blk * BLOCK * STRIDE - PADDING
    block_last = (pid_o_blk * BLOCK + BLOCK -
                  1) * STRIDE - PADDING + (K - 1) * DILATION
    full_tile = (pid_o_blk * BLOCK + BLOCK
                 <= L_out) & (block_start >= 0) & (block_last < L_in)

    y_max = tl.full((BLOCK, ), -float("inf"), tl.float32)
    if full_tile:
        for k in tl.static_range(0, K):
            xk = tl.load(base_x + (starts + k * DILATION).to(tl.int64),
                         mask=mask_o,
                         other=-float("inf")).to(tl.float32)
            y_max = tl.maximum(y_max, xk)
    else:
        for k in tl.static_range(0, K):
            pos = starts + k * DILATION
            valid = (pos >= 0) & (pos < L_in) & mask_o
            addr = tl.minimum(tl.maximum(pos, 0), L_in - 1)
            xk = tl.load(base_x + addr.to(tl.int64),
                         mask=valid,
                         other=-float("inf")).to(tl.float32)
            y_max = tl.maximum(y_max, xk)

    tl.store(base_y + o_offsets.to(tl.int64), y_max, mask=mask_o)


@triton.jit
def _maxpool1d_noindex_persistent_kernel(
    x_ptr,
    y_ptr,
    NC,
    L_in,
    L_out,
    STRIDE,
    PADDING,
    DILATION,
    line_stride_x,
    line_stride_y,
    n_o_blks,
    total_tiles,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    n_programs = tl.num_programs(axis=0)

    for tile in range(pid, total_tiles, n_programs):
        pid_nc = tile // n_o_blks
        pid_o_blk = tile - pid_nc * n_o_blks

        o_offsets = pid_o_blk * BLOCK + tl.arange(0, BLOCK)
        mask_o = o_offsets < L_out
        starts = o_offsets * STRIDE - PADDING

        base_x = x_ptr + pid_nc.to(tl.int64) * line_stride_x
        base_y = y_ptr + pid_nc.to(tl.int64) * line_stride_y

        tl.multiple_of(o_offsets, 16)
        tl.max_contiguous(o_offsets, BLOCK)

        block_start = pid_o_blk * BLOCK * STRIDE - PADDING
        block_last = (pid_o_blk * BLOCK + BLOCK -
                      1) * STRIDE - PADDING + (K - 1) * DILATION
        full_tile = (pid_o_blk * BLOCK + BLOCK
                     <= L_out) & (block_start >= 0) & (block_last < L_in)

        y_max = tl.full((BLOCK, ), -float("inf"), tl.float32)
        if full_tile:
            for k in tl.static_range(0, K):
                xk = tl.load(base_x + (starts + k * DILATION).to(tl.int64),
                             mask=mask_o,
                             other=-float("inf")).to(tl.float32)
                y_max = tl.maximum(y_max, xk)
        else:
            for k in tl.static_range(0, K):
                pos = starts + k * DILATION
                valid = (pos >= 0) & (pos < L_in) & mask_o
                addr = tl.minimum(tl.maximum(pos, 0), L_in - 1)
                xk = tl.load(base_x + addr.to(tl.int64),
                             mask=valid,
                             other=-float("inf")).to(tl.float32)
                y_max = tl.maximum(y_max, xk)

        tl.store(base_y + o_offsets.to(tl.int64), y_max, mask=mask_o)


@triton.jit
def _maxpool1d_index_kernel(
    x_ptr,
    y_ptr,
    idx_ptr,
    L_in,
    L_out,
    STRIDE,
    PADDING,
    DILATION,
    line_stride_x,
    line_stride_y,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid_nc = tl.program_id(axis=0)
    pid_o_blk = tl.program_id(axis=1)

    o_offsets = pid_o_blk * BLOCK + tl.arange(0, BLOCK)
    mask_o = o_offsets < L_out
    starts = o_offsets * STRIDE - PADDING

    base_x = x_ptr + pid_nc.to(tl.int64) * line_stride_x
    base_y = y_ptr + pid_nc.to(tl.int64) * line_stride_y
    base_i = idx_ptr + pid_nc.to(tl.int64) * line_stride_y

    pos = starts
    valid0 = (pos >= 0) & (pos < L_in) & mask_o
    addr0 = tl.minimum(tl.maximum(pos, 0), L_in - 1)
    x0 = tl.load(base_x + addr0.to(tl.int64), mask=mask_o,
                 other=0).to(tl.float32)
    y_max = tl.where(valid0, x0, -float("inf"))
    chosen_pos = pos

    for _ in tl.static_range(1, K):
        pos = pos + DILATION
        validk = (pos >= 0) & (pos < L_in) & mask_o
        addrk = tl.minimum(tl.maximum(pos, 0), L_in - 1)
        xk = tl.load(base_x + addrk.to(tl.int64), mask=mask_o,
                     other=0).to(tl.float32)
        vk = tl.where(validk, xk, -float("inf"))
        better = vk > y_max
        y_max = tl.where(better, vk, y_max)
        chosen_pos = tl.where(better, pos, chosen_pos)

    tl.store(base_y + o_offsets.to(tl.int64), y_max, mask=mask_o)
    chosen_pos = tl.maximum(0, tl.minimum(chosen_pos, L_in - 1))
    tl.store(base_i + o_offsets.to(tl.int64),
             chosen_pos.to(tl.int64),
             mask=mask_o)


class ModelNew(nn.Module):
    """MaxPool1d optimized for the default no-index large-output regime."""

    def __init__(
        self,
        kernel_size: int = DEFAULT_KERNEL_SIZE,
        stride: int = DEFAULT_STRIDE,
        padding: int = DEFAULT_PADDING,
        dilation: int = DEFAULT_DILATION,
        return_indices: bool = DEFAULT_RETURN_INDICES,
    ):
        super(ModelNew, self).__init__()
        self.kernel_size = int(kernel_size)
        self.stride = int(kernel_size if stride is None else stride)
        self.padding = int(padding)
        self.dilation = int(dilation)
        self.return_indices = bool(return_indices)

    def _out_length(self, L_in: int) -> int:
        return (L_in + 2 * self.padding - self.dilation *
                (self.kernel_size - 1) - 1) // self.stride + 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.device.type != "npu":
            raise ValueError("ModelNew expects Ascend NPU inputs")
        if x.dtype not in (torch.float16, torch.float32, torch.bfloat16):
            raise TypeError(
                "ModelNew supports float16, float32, and bfloat16 inputs only")

        x = x.contiguous()
        N, C, L_in = x.shape
        L_out = self._out_length(L_in)
        if L_out <= 0:
            raise ValueError(
                "Invalid pooling configuration produces a non-positive output length"
            )

        if self.return_indices:
            return F.max_pool1d(
                x,
                self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                ceil_mode=False,
                return_indices=True,
            )

        y = torch.empty((N, C, L_out), device=x.device, dtype=x.dtype)
        NC = N * C
        BLOCK = 128
        n_o_blks = triton.cdiv(L_out, BLOCK)
        total_tiles = NC * n_o_blks

        if total_tiles > _MAX_PROGRAMS:
            try:
                props = driver.active.utils.get_device_properties(
                    torch.npu.current_device())
                core_num = int(props.get("num_vectorcore", 40))
            except Exception:
                core_num = 40
            n_programs = max(1, min(core_num, _MAX_PROGRAMS, total_tiles))
            _maxpool1d_noindex_persistent_kernel[(n_programs, )](
                x,
                y,
                NC,
                L_in,
                L_out,
                self.stride,
                self.padding,
                self.dilation,
                L_in,
                L_out,
                n_o_blks,
                total_tiles,
                K=self.kernel_size,
                BLOCK=BLOCK,
                num_warps=4,
                num_stages=2,
            )
        else:
            _maxpool1d_noindex_direct_kernel[(NC, n_o_blks)](
                x,
                y,
                L_in,
                L_out,
                self.stride,
                self.padding,
                self.dilation,
                L_in,
                L_out,
                K=self.kernel_size,
                BLOCK=BLOCK,
                num_warps=4,
                num_stages=2,
            )
        return y


def max_pool1d(
    x: torch.Tensor,
    kernel_size: int = DEFAULT_KERNEL_SIZE,
    stride: int = DEFAULT_STRIDE,
    padding: int = DEFAULT_PADDING,
    dilation: int = DEFAULT_DILATION,
    return_indices: bool = DEFAULT_RETURN_INDICES,
):
    return ModelNew(
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        return_indices=return_indices,
    )(x)


batch_size = 64
features = 192
sequence_length = 65536
kernel_size = 8
stride = 1
padding = 4
dilation = 3
return_indices = False


def get_inputs():
    x = torch.rand(batch_size, features, sequence_length)
    return [x]


def get_init_inputs():
    return [kernel_size, stride, padding, dilation, return_indices]
