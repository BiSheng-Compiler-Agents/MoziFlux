import torch
import torch.nn as nn
import torch_npu  # noqa: F401
import triton
import triton.language as tl


# ── Direct kernel (one program per tile, no persistent loop) ────────────────────
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8, num_stages=2),
    ],
    key=["n_elements_pow2"],
)
@triton.jit
def _relu_kernel_direct(
    x_ptr,
    y_ptr,
    n_elements,
    n_elements_pow2: tl.
    constexpr,  # constexpr: autotune key, zero runtime load
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    tl.multiple_of(offsets, 16)
    tl.max_contiguous(offsets, 16)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0, care_padding=False)
    x_fp32 = x.to(tl.float32)
    y_fp32 = tl.maximum(x_fp32, 0.0, propagate_nan=tl.PropagateNan.ALL)
    y = y_fp32.to(x.dtype)
    tl.store(y_ptr + offsets, y, mask=mask)


# ── Persistent kernel (work-stealing loop) ──────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8, num_stages=2),
    ],
    key=["n_elements_pow2"],
)
@triton.jit
def _relu_kernel_persistent(
    x_ptr,
    y_ptr,
    n_elements,
    n_elements_pow2: tl.
    constexpr,  # constexpr: autotune key, zero runtime load
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    n_programs = tl.num_programs(0)
    tile_id = pid
    while tile_id * BLOCK_SIZE < n_elements:
        offsets = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        tl.multiple_of(offsets, 16)
        tl.max_contiguous(offsets, 16)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0, care_padding=False)
        x_fp32 = x.to(tl.float32)
        y_fp32 = tl.maximum(x_fp32, 0.0, propagate_nan=tl.PropagateNan.ALL)
        y = y_fp32.to(x.dtype)
        tl.store(y_ptr + offsets, y, mask=mask)
        tile_id += n_programs


def _next_pow2(n: int) -> int:
    """Smallest power-of-2 >= n. Bucketed autotune key."""
    return 1 << (n - 1).bit_length()


class ModelNew(nn.Module):
    """
    Optimized ReLU for Ascend NPU.

    Dispatch strategy:
      n <= 256*65535 (16.78M) : _relu_kernel_direct  — no loop overhead,
                                 guaranteed grid <= 65535 for all BLOCK_SIZE configs
      n >  256*65535           : _relu_kernel_persistent — while-loop covers all
                                 elements, grid capped at 65535

    n_elements_pow2 is passed as tl.constexpr so it is a compile-time constant
    and generates zero runtime SCALARLDST loads (unlike a plain i32 arg).
    """

    MAX_PROGRAMS = 65535

    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.device.type != "npu":
            raise ValueError("ModelNew expects an Ascend NPU tensor input")
        if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise TypeError(
                "ModelNew supports only float16, bfloat16, and float32 tensors"
            )
        if x.numel() == 0:
            return torch.empty_like(x)

        x_contig = x.contiguous()
        y = torch.empty_like(x_contig)
        x_flat = x_contig.view(-1)
        y_flat = y.view(-1)
        n = x_flat.numel()
        n_pow2 = _next_pow2(n)

        # Threshold: largest n where cdiv(n, 256) <= 65535
        # => n <= 256 * 65535 = 16,776,960
        # Guarantees every autotune BLOCK_SIZE config stays within FFTS limit
        # on the direct path.
        if triton.cdiv(n, 256) > self.MAX_PROGRAMS:

            def grid(meta):
                return (min(triton.cdiv(n, meta["BLOCK_SIZE"]),
                            self.MAX_PROGRAMS), )

            _relu_kernel_persistent[grid](x_flat, y_flat, n, n_pow2)
        else:

            def direct_grid(meta):
                return (triton.cdiv(n, meta["BLOCK_SIZE"]), )

            _relu_kernel_direct[direct_grid](x_flat, y_flat, n, n_pow2)

        return y.view_as(x)


# Benchmark shape from KernelBench
batch_size = 4096
dim = 393216


def get_inputs():
    x = torch.rand(batch_size, dim)
    return [x]


def get_init_inputs():
    return []
