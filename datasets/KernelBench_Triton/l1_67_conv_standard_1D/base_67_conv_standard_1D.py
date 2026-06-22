import importlib.util
import os as _os
import torch
import torch.nn as nn
import triton

_here = _os.path.dirname(_os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "baseline_kernels",
    _os.path.join(_here, "..", "baseline", "67_conv_standard_1D.py"))
_bl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bl)
conv1d_fwd_kernel = _bl.conv1d_fwd_kernel


def conv1d_triton_fp32(x: torch.Tensor,
                       w: torch.Tensor,
                       stride: int = 1,
                       padding: int = 0,
                       dilation: int = 1) -> torch.Tensor:
    B, C, L_IN = x.shape
    OC, Cw, K = w.shape
    assert Cw == C
    L_OUT = (L_IN + 2 * padding - dilation * (K - 1) - 1) // stride + 1
    x = x.contiguous() if not x.is_contiguous() else x
    w = w.contiguous() if not w.is_contiguous() else w
    y = torch.empty((B, OC, L_OUT), device=x.device, dtype=torch.float32)
    CK = C * K
    if CK <= 16:
        bp, bt, boc, nw, ns = 16, 128, (64 if OC >= 64 else 32), 4, 3
    elif CK <= 64:
        bp, bt, boc, nw, ns = 32, 128, (64 if OC >= 64 else
                                        32), (8 if OC >= 128 else 4), 3
    else:
        bp, bt, boc, nw, ns = 64, 128, (64 if OC >= 64 else
                                        32), (8 if OC >= 128 else 4), 4
    npi = (CK + bp - 1) // bp
    grid = (triton.cdiv(L_OUT, bt), triton.cdiv(OC, boc), B)
    conv1d_fwd_kernel[grid](x,
                            w,
                            y,
                            B,
                            C,
                            L_IN,
                            OC,
                            K,
                            stride,
                            padding,
                            dilation,
                            L_OUT,
                            CK,
                            BLOCK_OC=boc,
                            BLOCK_T=bt,
                            BLOCK_P=bp,
                            NUM_P_ITERS=npi,
                            num_warps=nw,
                            num_stages=ns)
    return y


def _conv1d_triton_fp32(x, w, stride, padding, dilation):
    return conv1d_triton_fp32(x,
                              w,
                              stride=stride,
                              padding=padding,
                              dilation=dilation)


class ModelNew(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=False):
        super().__init__()
        self.conv1d = nn.Conv1d(in_channels,
                                out_channels,
                                kernel_size,
                                stride=stride,
                                padding=padding,
                                dilation=dilation,
                                groups=groups,
                                bias=bias)

    def forward(self, x):
        if self.conv1d.groups != 1:
            raise NotImplementedError()
        if self.conv1d.bias is not None:
            raise NotImplementedError()
        s = self.conv1d.stride[0] if isinstance(self.conv1d.stride,
                                                tuple) else self.conv1d.stride
        p = self.conv1d.padding[0] if isinstance(
            self.conv1d.padding, tuple) else self.conv1d.padding
        d = self.conv1d.dilation[0] if isinstance(
            self.conv1d.dilation, tuple) else self.conv1d.dilation
        return conv1d_triton_fp32(x,
                                  self.conv1d.weight,
                                  stride=s,
                                  padding=p,
                                  dilation=d)


batch_size = 32
in_channels = 64
out_channels = 128
kernel_size = 3
length = 131072


def get_inputs():
    x = torch.rand(batch_size, in_channels, length)
    return [x]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size]
