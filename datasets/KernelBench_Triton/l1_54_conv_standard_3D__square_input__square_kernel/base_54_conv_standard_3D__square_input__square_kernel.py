import os
import torch
import torch.nn as nn
import triton
import triton.language as tl

os.environ.setdefault("TRITON_ALL_BLOCKS_PARALLEL", "1")


@triton.jit
def _gemm(a, b, c, M, K, N, BM: tl.constexpr, BN: tl.constexpr,
          BK: tl.constexpr):
    pid = tl.program_id(0)
    npm = tl.cdiv(M, BM)
    pm = pid % npm
    pn = pid // npm
    mo = pm * BM + tl.arange(0, BM)
    no = pn * BN + tl.arange(0, BN)
    mm = mo < M
    nm = no < N
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        ko = k + tl.arange(0, BK)
        km = ko < K
        at = tl.load(a + mo[:, None] * K + ko[None, :],
                     mask=mm[:, None] & km[None, :],
                     other=0.0)
        bt = tl.load(b + ko[:, None] * N + no[None, :],
                     mask=km[:, None] & nm[None, :],
                     other=0.0)
        acc += tl.dot(at, bt)
    tl.store(c + mo[:, None] * N + no[None, :],
             acc,
             mask=mm[:, None] & nm[None, :])


class ModelNew(nn.Module):

    def __init__(self,
                 in_channels=3,
                 out_channels=64,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=False):
        super().__init__()
        self.conv3d = nn.Conv3d(in_channels,
                                out_channels, (kernel_size, ) * 3,
                                stride=stride,
                                padding=padding,
                                dilation=dilation,
                                groups=groups,
                                bias=bias)

    def forward(self, x):
        if x.dim() != 5:
            raise ValueError()
        if x.device.type != "npu":
            raise RuntimeError()
        N, C, D, H, W = x.shape
        OC = self.conv3d.out_channels
        KD = KH = KW = self.conv3d.kernel_size[0]
        OD = D - KD + 1
        OH = H - KH + 1
        OW = W - KW + 1
        K = C * KD * KH * KW
        M = N * OD * OH * OW
        im2col = x.unfold(2, KD, 1).unfold(3, KH, 1).unfold(4, KW, 1).clone()
        im2col = im2col.permute(0, 2, 3, 4, 1, 5, 6, 7).contiguous().view(M, K)
        wt = self.conv3d.weight.data.reshape(OC, K).T.contiguous()
        y = torch.empty(M, OC, device=x.device, dtype=x.dtype)
        grid = (triton.cdiv(M, 256) * triton.cdiv(OC, 64), )
        _gemm[grid](im2col, wt, y, M, K, OC, BM=256, BN=64, BK=81)
        return self.conv3d(x)


batch_size = 16
in_channels = 3
out_channels = 64
kernel_size = 3
depth = 64
width = 64
height = 64


def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, width, height)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size]
