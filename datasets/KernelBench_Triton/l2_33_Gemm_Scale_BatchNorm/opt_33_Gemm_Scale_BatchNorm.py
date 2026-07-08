import torch
import torch.nn as nn
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al

try:
    import torch_npu  # noqa: F401
except ImportError:
    torch_npu = None


@triton.jit
def _linear_scale_kernel_opt(
    A_ptr,  # [M, K]
    B_ptr,  # [K, N] contiguous cached transpose of nn.Linear weight
    Bias_ptr,  # [N]
    Scale_ptr,  # [N]
    C_ptr,  # [M, N] fp32
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    # Grouped 1D scheduling improves B-tile/L2 reuse and avoids multidim-grid overhead.
    GROUP_M: tl.constexpr = 4
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_in_group = pid % num_pid_in_group
    pid_m = first_pid_m + (pid_in_group % group_size_m)
    pid_n = pid_in_group // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    tl.multiple_of(offs_m, 16)
    tl.multiple_of(offs_n, 16)
    tl.multiple_of(offs_k, 16)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    al.compile_hint(acc, "dot_pad_only_k")

    for k_start in tl.range(0, K, BLOCK_K):
        a_ptrs = A_ptr + offs_m[:, None] * stride_am + (
            k_start + offs_k[None, :]) * stride_ak
        b_ptrs = B_ptr + (k_start + offs_k[:, None]
                          ) * stride_bk + offs_n[None, :] * stride_bn
        a = tl.load(a_ptrs,
                    mask=(offs_m[:, None] < M) &
                    ((k_start + offs_k[None, :]) < K),
                    other=0.0)
        b = tl.load(b_ptrs,
                    mask=((k_start + offs_k[:, None]) < K) &
                    (offs_n[None, :] < N),
                    other=0.0)
        acc = tl.dot(a, b, acc)

    bias = tl.load(Bias_ptr + offs_n, mask=offs_n < N,
                   other=0.0).to(tl.float32)
    scale = tl.load(Scale_ptr + offs_n, mask=offs_n < N,
                    other=1.0).to(tl.float32)
    acc = (acc + bias[None, :]) * scale[None, :]

    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


class ModelNew(nn.Module):
    """GEMM + scale + BatchNorm1d with an optimized Triton GEMM/scale stage."""

    def __init__(
        self,
        in_features=1024,
        out_features=512,
        scale_shape=None,
        eps=1e-5,
        momentum=0.1,
        device="npu",
        dtype=torch.float32,
    ):
        super(ModelNew, self).__init__()
        if scale_shape is None:
            scale_shape = (out_features, )
        self.gemm = nn.Linear(in_features,
                              out_features,
                              device=device,
                              dtype=dtype)
        self.scale = nn.Parameter(
            torch.randn(scale_shape, device=device, dtype=dtype))
        self.bn = nn.BatchNorm1d(out_features,
                                 eps=eps,
                                 momentum=momentum,
                                 device=device,
                                 dtype=dtype)
        self._cached_weight_kn = None
        self._cached_weight_key = None

    def _ensure_device_dtype(self, x: torch.Tensor) -> None:
        param = next(self.parameters())
        if param.device != x.device or param.dtype != x.dtype:
            self.to(device=x.device, dtype=x.dtype)
            self._cached_weight_kn = None
            self._cached_weight_key = None

    def _weight_kn(self) -> torch.Tensor:
        w = self.gemm.weight
        key = (w.data_ptr(), tuple(w.shape), w.dtype, w.device,
               getattr(w, "_version", 0))
        if self._cached_weight_key != key or self._cached_weight_kn is None:
            # nn.Linear stores [N, K]; the kernel consumes [K, N] so B loads are contiguous.
            self._cached_weight_kn = w.transpose(0, 1).contiguous()
            self._cached_weight_key = key
        return self._cached_weight_kn

    def _fused_linear_scale(self, x: torch.Tensor) -> torch.Tensor:
        M, K = x.shape
        N = self.gemm.weight.shape[0]
        A = x.contiguous()
        B = self._weight_kn()
        Bias = self.gemm.bias.contiguous(
        ) if self.gemm.bias is not None else torch.zeros(
            N, device=x.device, dtype=x.dtype)
        Scale = self.scale.contiguous()
        y = torch.empty((M, N), device=x.device, dtype=torch.float32)

        def grid(META):
            return (triton.cdiv(M, META["BLOCK_M"]) *
                    triton.cdiv(N, META["BLOCK_N"]), )

        _linear_scale_kernel_opt[grid](
            A,
            B,
            Bias,
            Scale,
            y,
            M,
            N,
            K,
            A.stride(0),
            A.stride(1),
            B.stride(0),
            B.stride(1),
            y.stride(0),
            y.stride(1),
            BLOCK_M=128,
            BLOCK_N=128,
            BLOCK_K=32,
        )
        return y

    def forward(self, x):
        if x.device.type != "npu":
            raise RuntimeError("ModelNew expects NPU inputs.")
        self._ensure_device_dtype(x)
        y = self._fused_linear_scale(x)
        return self.bn(y)


batch_size = 1024
in_features = 8192
out_features = 8192
scale_shape = (out_features, )


def get_inputs():
    return [torch.rand(batch_size, in_features)]


def get_init_inputs():
    return [in_features, out_features, scale_shape]
