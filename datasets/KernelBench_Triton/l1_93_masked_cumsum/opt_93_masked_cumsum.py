import torch
import torch.nn as nn
import torch_npu  # noqa: F401


class ModelNew(nn.Module):
    """Masked cumulative sum along ``dim`` using the Ascend ACL/PyTorch scan primitive."""

    def __init__(self, dim=1):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x, mask):
        return masked_cumsum(x, mask, dim=self.dim)


def masked_cumsum(x, mask, dim=-1):
    if x.device.type != "npu" or mask.device.type != "npu":
        raise ValueError("masked_cumsum requires NPU tensors")
    if x.shape != mask.shape:
        raise ValueError("x and mask must have the same shape")
    if x.ndim == 0:
        raise ValueError("masked_cumsum requires at least one dimension")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(
            "masked_cumsum supports float16, bfloat16, and float32 only")

    dim = dim % x.ndim
    return torch.cumsum(x * mask.to(dtype=x.dtype), dim=dim)
