import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

batch_size = 128
in_channels = 8
out_channels = 64
height, width = 128, 128
kernel_size = 3
scaling_factor = 2.0

_MAX_PROGRAMS = 65535
_SCALE_BLOCK = 4096


@triton.jit
def _scale_direct_kernel(x_ptr, y_ptr, scale, n_elements,
                         BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0, care_padding=False)
    y = x * scale
    tl.store(y_ptr + offs, y, mask=mask)


@triton.jit
def _scale_persistent_kernel(x_ptr, y_ptr, scale, n_elements, n_programs,
                             BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    n_tiles = tl.cdiv(n_elements, BLOCK_SIZE)
    for tile_id in range(pid, n_tiles, n_programs):
        offs = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements
        x = tl.load(x_ptr + offs, mask=mask, other=0.0, care_padding=False)
        y = x * scale
        tl.store(y_ptr + offs, y, mask=mask)


def _scale_triton(x: torch.Tensor,
                  scale: float,
                  force_persistent: bool = False) -> torch.Tensor:
    if x.device.type != "npu":
        raise RuntimeError("_scale_triton expects an Ascend NPU tensor input")
    x_contig = x.contiguous()
    y = torch.empty_like(x_contig)
    n_elements = x_contig.numel()
    if n_elements == 0:
        return y
    n_tiles = triton.cdiv(n_elements, _SCALE_BLOCK)
    if force_persistent or n_tiles > _MAX_PROGRAMS:
        grid = (min(n_tiles, _MAX_PROGRAMS), )
        _scale_persistent_kernel[grid](x_contig,
                                       y,
                                       float(scale),
                                       n_elements,
                                       grid[0],
                                       BLOCK_SIZE=_SCALE_BLOCK,
                                       num_warps=4,
                                       num_stages=2)
    else:
        _scale_direct_kernel[(n_tiles, )](x_contig,
                                          y,
                                          float(scale),
                                          n_elements,
                                          BLOCK_SIZE=_SCALE_BLOCK,
                                          num_warps=4,
                                          num_stages=2)
    return y


class ModelNew(nn.Module):
    """Conv2d -> BatchNorm2d -> scale with the scale fused into BN parameters.

    Training/non-tracked BN uses native ACL conv and batch_norm with scaled affine
    parameters, avoiding a full-output Triton scaling pass. Eval mode caches the
    Conv+BN+scale folded weights/bias until any source tensor version changes.
    """

    def __init__(self,
                 in_channels=in_channels,
                 out_channels=out_channels,
                 kernel_size=kernel_size,
                 scaling_factor=scaling_factor):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels)
        self.scaling_factor = scaling_factor
        self._eval_cache_key = None
        self._eval_cache_value = None
        self._bn_affine_cache_key = None
        self._bn_affine_cache_value = None

    @staticmethod
    def _tensor_key(t):
        if t is None:
            return (0, -1)
        return (int(t.data_ptr()), int(getattr(t, "_version", 0)))

    def _get_eval_fused_weight_bias(self):
        W = self.conv.weight
        B = self.conv.bias
        dtype = W.dtype
        C = self.bn.num_features
        gamma = self.bn.weight if self.bn.affine else None
        beta = self.bn.bias if self.bn.affine else None
        key = (
            self._tensor_key(W),
            self._tensor_key(B),
            self._tensor_key(gamma),
            self._tensor_key(beta),
            self._tensor_key(self.bn.running_mean),
            self._tensor_key(self.bn.running_var),
            float(self.bn.eps),
            float(self.scaling_factor),
            dtype,
            W.device,
        )
        if key == self._eval_cache_key and self._eval_cache_value is not None:
            return self._eval_cache_value

        if gamma is None:
            gamma_t = torch.ones(C, device=W.device, dtype=dtype)
            beta_t = torch.zeros(C, device=W.device, dtype=dtype)
        else:
            gamma_t = gamma.to(device=W.device, dtype=dtype)
            beta_t = beta.to(device=W.device, dtype=dtype)
        mean = self.bn.running_mean.to(device=W.device, dtype=dtype)
        var = self.bn.running_var.to(device=W.device, dtype=dtype)
        conv_bias = B if B is not None else torch.zeros(
            C, device=W.device, dtype=dtype)
        inv_std = torch.rsqrt(var + self.bn.eps)
        g = gamma_t * float(self.scaling_factor) * inv_std
        b = beta_t * float(self.scaling_factor) + (conv_bias - mean) * g
        fused = (W * g.view(-1, 1, 1, 1), b)
        self._eval_cache_key = key
        self._eval_cache_value = fused
        return fused

    def _get_scaled_bn_affine(self, x):
        s = float(self.scaling_factor)
        C = self.bn.num_features
        if torch.is_grad_enabled():
            if self.bn.affine:
                return self.bn.weight * s, self.bn.bias * s
            return (torch.full((C, ), s, device=x.device, dtype=x.dtype),
                    torch.zeros((C, ), device=x.device, dtype=x.dtype))
        key = (
            self._tensor_key(self.bn.weight if self.bn.affine else None),
            self._tensor_key(self.bn.bias if self.bn.affine else None),
            s,
            C,
            x.dtype,
            x.device,
        )
        if key == self._bn_affine_cache_key and self._bn_affine_cache_value is not None:
            return self._bn_affine_cache_value
        if self.bn.affine:
            value = (self.bn.weight * s, self.bn.bias * s)
        else:
            value = (torch.full((C, ), s, device=x.device, dtype=x.dtype),
                     torch.zeros((C, ), device=x.device, dtype=x.dtype))
        self._bn_affine_cache_key = key
        self._bn_affine_cache_value = value
        return value

    def forward(self, x):
        if (not self.bn.training) and self.bn.track_running_stats:
            W_fused, b_fused = self._get_eval_fused_weight_bias()
            return F.conv2d(x,
                            W_fused,
                            b_fused,
                            stride=self.conv.stride,
                            padding=self.conv.padding,
                            dilation=self.conv.dilation,
                            groups=self.conv.groups)

        x = self.conv(x)
        training_flag = self.bn.training or not self.bn.track_running_stats
        running_mean = self.bn.running_mean if self.bn.track_running_stats else None
        running_var = self.bn.running_var if self.bn.track_running_stats else None
        weight, bias = self._get_scaled_bn_affine(x)
        momentum = self.bn.momentum if self.bn.momentum is not None else 0.0
        return F.batch_norm(x,
                            running_mean=running_mean,
                            running_var=running_var,
                            weight=weight,
                            bias=bias,
                            training=training_flag,
                            momentum=momentum,
                            eps=self.bn.eps)


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, scaling_factor]
