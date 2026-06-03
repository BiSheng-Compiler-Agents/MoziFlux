import torch
import torch.nn as nn
import torch_npu  # noqa: F401
import triton
import triton.language as tl


@triton.jit
def _elu_kernel(x_ptr, y_ptr, N, alpha, NUM_BLOCKS, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    base = pid * BLOCK_SIZE
    step = NUM_BLOCKS * BLOCK_SIZE
    inv_ln2 = 1.4426950408889634

    while base < N:
        offs = base + tl.arange(0, BLOCK_SIZE)
        mask = offs < N
        tl.max_contiguous(offs, 128)
        x = tl.load(x_ptr + offs, mask=mask, other=0)
        x_pos = tl.maximum(x, 0)
        x_neg = tl.minimum(x, 0)
        exp_term = tl.exp2(x_neg * inv_ln2)
        y = x_pos + (exp_term - 1) * alpha
        tl.store(y_ptr + offs, y, mask=mask)
        base += step


class ModelNew(nn.Module):
    def __init__(self, alpha: float = 1.0):
        super(ModelNew, self).__init__()
        self.alpha = float(alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.device.type != "npu":
            raise ValueError("ModelNew expects an Ascend NPU tensor input")
        if x.requires_grad:
            raise ValueError("ModelNew does not support autograd-enabled inputs")
        if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise TypeError(
                "ModelNew supports only float16, bfloat16, and float32 tensors"
            )
        if x.numel() == 0:
            return torch.empty_like(x)

        x_contig = x.contiguous()
        y = torch.empty_like(x_contig)
        N = x_contig.numel()

        if N >= 131072:
            BLOCK_SIZE = 8192
            num_warps = 8
        elif N >= 32768:
            BLOCK_SIZE = 4096
            num_warps = 8
        elif N >= 8192:
            BLOCK_SIZE = 2048
            num_warps = 4
        else:
            BLOCK_SIZE = 1024
            num_warps = 4

        total_tiles = triton.cdiv(N, BLOCK_SIZE)
        num_blocks = min(total_tiles, 5120)

        def grid(meta):
            return (num_blocks,)

        _elu_kernel[grid](
            x_contig.view(-1),
            y.view(-1),
            N,
            self.alpha,
            num_blocks,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
            num_stages=2,
        )
        return y.view_as(x)


batch_size = 4096
dim = 393216


def get_inputs():
    x = torch.rand(batch_size, dim)
    return [x]


def get_init_inputs():
    return [1.0]
