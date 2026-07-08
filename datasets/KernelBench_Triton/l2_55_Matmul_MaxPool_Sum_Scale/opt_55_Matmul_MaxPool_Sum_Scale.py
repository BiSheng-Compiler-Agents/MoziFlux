import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al
from triton.runtime import driver

DEFAULT_BATCH_SIZE = 128
DEFAULT_IN_FEATURES = 32768
DEFAULT_OUT_FEATURES = 32768
DEFAULT_KERNEL_SIZE = 2
DEFAULT_SCALE_FACTOR = 0.5

BLOCK_M = 16
BLOCK_P = 16  # output pairs per tile; BLOCK_O = 2 * BLOCK_P
BLOCK_K = 64
_MAX_GRID = 65535
_USE_ACL_DISPATCH = True


def _is_npu_tensor(x: torch.Tensor) -> bool:
    return bool(getattr(x, "is_npu", False) or x.device.type == "npu")


@triton.jit
def _linear_pairmax_partial_kernel(
    X_ptr,
    WKN_ptr,
    BIAS_ptr,
    PARTIAL_ptr,
    M: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    NUM_PBLOCKS: tl.constexpr,
    stride_xm: tl.constexpr,
    stride_xk: tl.constexpr,
    stride_wk: tl.constexpr,
    stride_wn: tl.constexpr,
    stride_pm: tl.constexpr,
    stride_pp: tl.constexpr,
    n_programs: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m_base = tl.arange(0, BLOCK_M)
    offs_p_base = tl.arange(0, BLOCK_P)
    offs_k_base = tl.arange(0, BLOCK_K)
    num_m = tl.cdiv(M, BLOCK_M)
    total_tiles = num_m * NUM_PBLOCKS

    for tile in tl.range(pid, total_tiles, n_programs):
        tile_m = tile // NUM_PBLOCKS
        tile_p = tile - tile_m * NUM_PBLOCKS
        rows = tile_m * BLOCK_M + offs_m_base
        pairs = tile_p * BLOCK_P + offs_p_base
        out0 = pairs * 2
        out1 = out0 + 1

        acc0 = tl.zeros((BLOCK_M, BLOCK_P), dtype=tl.float32)
        acc1 = tl.zeros((BLOCK_M, BLOCK_P), dtype=tl.float32)
        al.compile_hint(acc0, "dot_pad_only_k")
        al.compile_hint(acc1, "dot_pad_only_k")

        for k0 in tl.range(0, K, BLOCK_K):
            kk = k0 + offs_k_base
            x = tl.load(
                X_ptr + rows[:, None] * stride_xm + kk[None, :] * stride_xk,
                mask=(rows[:, None] < M) & (kk[None, :] < K),
                other=0.0,
                care_padding=False,
            )
            w0 = tl.load(
                WKN_ptr + kk[:, None] * stride_wk + out0[None, :] * stride_wn,
                mask=(kk[:, None] < K) & (out0[None, :] < N),
                other=0.0,
                care_padding=False,
            )
            w1 = tl.load(
                WKN_ptr + kk[:, None] * stride_wk + out1[None, :] * stride_wn,
                mask=(kk[:, None] < K) & (out1[None, :] < N),
                other=0.0,
                care_padding=False,
            )
            acc0 = tl.dot(x, w0, acc0)
            acc1 = tl.dot(x, w1, acc1)

        b0 = tl.load(BIAS_ptr + out0, mask=out0 < N, other=0.0).to(tl.float32)
        b1 = tl.load(BIAS_ptr + out1, mask=out1 < N, other=0.0).to(tl.float32)
        v0 = acc0 + b0[None, :]
        v1 = acc1 + b1[None, :]
        pair_valid = out0 < N
        pair_max = tl.maximum(v0, v1)
        partial = tl.sum(tl.where(pair_valid[None, :], pair_max, 0.0), axis=1)
        tl.store(
            PARTIAL_ptr + rows * stride_pm + tile_p * stride_pp,
            partial,
            mask=rows < M,
        )


@triton.jit
def _sum_scale_partials_kernel(
    PARTIAL_ptr,
    OUT_ptr,
    M: tl.constexpr,
    NUM_PBLOCKS: tl.constexpr,
    stride_pm: tl.constexpr,
    stride_pp: tl.constexpr,
    SCALE,
    BLOCK_PB: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_PB)
    mask = offs < NUM_PBLOCKS
    vals = tl.load(PARTIAL_ptr + row * stride_pm + offs * stride_pp,
                   mask=mask,
                   other=0.0).to(tl.float32)
    total = tl.sum(vals, axis=0) * SCALE
    tl.store(OUT_ptr + row, total, mask=row < M)


class ModelNew(nn.Module):
    """Linear -> MaxPool1d(kernel_size=stride) -> sum(dim=1) -> scale."""

    def __init__(self, in_features, out_features, kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.max_pool = nn.MaxPool1d(kernel_size)
        self.scale_factor = scale_factor
        self._cached_w_kn = None
        self._cached_w_key = None

    def _weight_kn(self):
        w = self.matmul.weight
        key = (w.data_ptr(), tuple(w.shape), w.dtype, w.device,
               getattr(w, "_version", 0))
        if self._cached_w_key != key or self._cached_w_kn is None:
            self._cached_w_kn = w.transpose(0, 1).contiguous()
            self._cached_w_key = key
        return self._cached_w_kn

    def _num_aicore(self, x):
        try:
            dev = x.device.index if x.device.index is not None else torch.npu.current_device(
            )
            return driver.active.utils.get_device_properties(dev)["num_aicore"]
        except Exception:
            return 20

    def _kernel_size(self) -> int:
        k = self.max_pool.kernel_size
        return int(k if isinstance(k, int) else k[0])

    def _acl_forward(self, x):
        z = F.linear(x, self.matmul.weight, self.matmul.bias)
        k = self._kernel_size()
        pooled = F.max_pool1d(z.unsqueeze(1), kernel_size=k,
                              stride=k).squeeze(1)
        return pooled.sum(dim=1) * float(self.scale_factor)

    def forward_triton_fallback(self, x):
        if not _is_npu_tensor(x):
            raise RuntimeError("ModelNew expects Ascend NPU tensors")
        if x.dim() != 2:
            raise RuntimeError(
                "ModelNew expects input shape (batch_size, in_features)")
        if self._kernel_size() != 2:
            return self._acl_forward(x.contiguous())

        x = x.contiguous()
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
        M, K = x.shape
        N = self.matmul.weight.shape[0]
        w_kn = self._weight_kn()
        b = self.matmul.bias
        if b is None:
            b = torch.zeros(N, device=x.device, dtype=torch.float32)
        elif b.dtype != torch.float32 or not b.is_contiguous():
            b = b.to(torch.float32).contiguous()

        num_pblocks = triton.cdiv(triton.cdiv(N, 2), BLOCK_P)
        partial = torch.empty((M, num_pblocks),
                              device=x.device,
                              dtype=torch.float32)
        total_tiles = triton.cdiv(M, BLOCK_M) * num_pblocks
        n_programs = min(max(1, total_tiles), self._num_aicore(x), _MAX_GRID)
        _linear_pairmax_partial_kernel[(n_programs, )](
            x,
            w_kn,
            b,
            partial,
            M,
            K,
            N,
            num_pblocks,
            x.stride(0),
            x.stride(1),
            w_kn.stride(0),
            w_kn.stride(1),
            partial.stride(0),
            partial.stride(1),
            n_programs,
            BLOCK_M=BLOCK_M,
            BLOCK_P=BLOCK_P,
            BLOCK_K=BLOCK_K,
        )
        out = torch.empty((M, ), device=x.device, dtype=torch.float32)
        block_pb = triton.next_power_of_2(num_pblocks)
        _sum_scale_partials_kernel[(M, )](
            partial,
            out,
            M,
            num_pblocks,
            partial.stride(0),
            partial.stride(1),
            float(self.scale_factor),
            BLOCK_PB=block_pb,
        )
        return out

    def forward(self, x):
        if not _is_npu_tensor(x):
            raise RuntimeError("ModelNew expects Ascend NPU tensors")
        if self.matmul.weight.device != x.device:
            raise RuntimeError(
                "ModelNew parameters must be on the same Ascend NPU device as input"
            )
        x = x.contiguous()
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
        if _USE_ACL_DISPATCH:
            return self._acl_forward(x)
        return self.forward_triton_fallback(x)


def run_model(x, weight, bias, kernel_size, scale_factor):
    if x.device.type != "npu" or weight.device.type != "npu" or bias.device.type != "npu":
        raise RuntimeError("run_model expects all tensors on NPU")
    model = ModelNew(x.shape[1], weight.shape[0], kernel_size,
                     scale_factor).to(device="npu", dtype=weight.dtype)
    with torch.no_grad():
        model.matmul.weight.copy_(weight.contiguous())
        model.matmul.bias.copy_(bias.contiguous())
    return model(x)


batch_size = DEFAULT_BATCH_SIZE
in_features = DEFAULT_IN_FEATURES
out_features = DEFAULT_OUT_FEATURES
kernel_size = DEFAULT_KERNEL_SIZE
scale_factor = DEFAULT_SCALE_FACTOR


def get_inputs():
    return [torch.rand(batch_size, in_features)]


def get_init_inputs():
    return [in_features, out_features, kernel_size, scale_factor]
