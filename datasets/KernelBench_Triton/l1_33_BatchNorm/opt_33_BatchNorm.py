import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _bn_reduce_hw_block(
    x_ptr,
    partial_sum_ptr,
    partial_sumsq_ptr,
    N: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    stride_n: tl.constexpr,
    stride_c: tl.constexpr,
    NUM_HW_BLOCKS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    c = tl.program_id(0)
    nb = tl.program_id(1)
    n = nb // NUM_HW_BLOCKS
    hw_block = nb - n * NUM_HW_BLOCKS
    offs = hw_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    hw = H * W
    mask = offs < hw
    vals = tl.load(x_ptr + n * stride_n + c * stride_c + offs,
                   mask=mask,
                   other=0.0).to(tl.float32)
    ss = tl.sum(vals, axis=0)
    sq = tl.sum(vals * vals, axis=0)
    out_idx = c * (N * NUM_HW_BLOCKS) + nb
    tl.store(partial_sum_ptr + out_idx, ss, mask=True)
    tl.store(partial_sumsq_ptr + out_idx, sq, mask=True)


@triton.jit
def _bn_reduce_hw_block_persistent(
    x_ptr,
    partial_sum_ptr,
    partial_sumsq_ptr,
    n_programs,
    TOTAL_TILES: tl.constexpr,
    N: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    stride_n: tl.constexpr,
    stride_c: tl.constexpr,
    NUM_HW_BLOCKS: tl.constexpr,
    NUM_PARTS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    for tile in range(pid, TOTAL_TILES, n_programs):
        c = tile // NUM_PARTS
        nb = tile - c * NUM_PARTS
        n = nb // NUM_HW_BLOCKS
        hw_block = nb - n * NUM_HW_BLOCKS
        offs = hw_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        hw = H * W
        mask = offs < hw
        vals = tl.load(x_ptr + n * stride_n + c * stride_c + offs,
                       mask=mask,
                       other=0.0).to(tl.float32)
        ss = tl.sum(vals, axis=0)
        sq = tl.sum(vals * vals, axis=0)
        out_idx = c * NUM_PARTS + nb
        tl.store(partial_sum_ptr + out_idx, ss, mask=True)
        tl.store(partial_sumsq_ptr + out_idx, sq, mask=True)


@triton.jit
def _bn_finalize_params_opt(
    partial_sum_ptr,
    partial_sumsq_ptr,
    scale_ptr,
    shift_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    NUM_PARTS: tl.constexpr,
    M: tl.constexpr,
    eps,
    exp_avg_factor,
    use_batch_stats: tl.constexpr,
    do_update: tl.constexpr,
    affine_flag: tl.constexpr,
    BLOCK_PARTS: tl.constexpr,
    NUM_PART_CHUNKS: tl.constexpr,
):
    c = tl.program_id(0)
    if use_batch_stats:
        offs = tl.arange(0, BLOCK_PARTS)
        acc_sum = tl.zeros((BLOCK_PARTS, ), dtype=tl.float32)
        acc_sumsq = tl.zeros((BLOCK_PARTS, ), dtype=tl.float32)
        for chunk in tl.static_range(0, NUM_PART_CHUNKS):
            idx = chunk * BLOCK_PARTS + offs
            mask = idx < NUM_PARTS
            base = c * NUM_PARTS + idx
            acc_sum += tl.load(partial_sum_ptr + base, mask=mask, other=0.0)
            acc_sumsq += tl.load(partial_sumsq_ptr + base,
                                 mask=mask,
                                 other=0.0)
        sum_v = tl.sum(acc_sum, axis=0)
        sumsq_v = tl.sum(acc_sumsq, axis=0)
        mean = sum_v / M
        var = sumsq_v / M - mean * mean
        var = tl.maximum(var, 0.0)
        invstd = 1.0 / tl.sqrt(var + eps)
        if do_update:
            rm = tl.load(running_mean_ptr + c, mask=True, other=0.0)
            rv = tl.load(running_var_ptr + c, mask=True, other=1.0)
            one_minus = 1.0 - exp_avg_factor
            unbiased_var = var
            if M > 1:
                unbiased_var = var * (M / (M - 1))
            tl.store(running_mean_ptr + c,
                     rm * one_minus + mean * exp_avg_factor,
                     mask=True)
            tl.store(running_var_ptr + c,
                     rv * one_minus + unbiased_var * exp_avg_factor,
                     mask=True)
        mean_use = mean
        invstd_use = invstd
    else:
        mean_use = tl.load(running_mean_ptr + c, mask=True, other=0.0)
        rv = tl.load(running_var_ptr + c, mask=True, other=1.0)
        invstd_use = 1.0 / tl.sqrt(rv + eps)

    if affine_flag:
        w = tl.load(weight_ptr + c, mask=True, other=1.0)
        b = tl.load(bias_ptr + c, mask=True, other=0.0)
        scale = invstd_use * w
        shift = b - mean_use * scale
    else:
        scale = invstd_use
        shift = -mean_use * invstd_use
    tl.store(scale_ptr + c, scale, mask=True)
    tl.store(shift_ptr + c, shift, mask=True)


@triton.jit
def _bn_apply_hw_block(
    x_ptr,
    y_ptr,
    scale_ptr,
    shift_ptr,
    N: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    stride_n: tl.constexpr,
    stride_c: tl.constexpr,
    NUM_HW_BLOCKS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    c = tl.program_id(0)
    nb = tl.program_id(1)
    n = nb // NUM_HW_BLOCKS
    hw_block = nb - n * NUM_HW_BLOCKS
    offs = hw_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    hw = H * W
    mask = offs < hw
    scale_c = tl.load(scale_ptr + c, mask=True, other=1.0)
    shift_c = tl.load(shift_ptr + c, mask=True, other=0.0)
    ptr = n * stride_n + c * stride_c + offs
    x = tl.load(x_ptr + ptr, mask=mask, other=0.0)
    y = x * scale_c + shift_c
    tl.store(y_ptr + ptr, y, mask=mask)


@triton.jit
def _bn_apply_hw_block_persistent(
    x_ptr,
    y_ptr,
    scale_ptr,
    shift_ptr,
    n_programs,
    TOTAL_TILES: tl.constexpr,
    N: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    stride_n: tl.constexpr,
    stride_c: tl.constexpr,
    NUM_HW_BLOCKS: tl.constexpr,
    NUM_PARTS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    for tile in range(pid, TOTAL_TILES, n_programs):
        c = tile // NUM_PARTS
        nb = tile - c * NUM_PARTS
        n = nb // NUM_HW_BLOCKS
        hw_block = nb - n * NUM_HW_BLOCKS
        offs = hw_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        hw = H * W
        mask = offs < hw
        scale_c = tl.load(scale_ptr + c, mask=True, other=1.0)
        shift_c = tl.load(shift_ptr + c, mask=True, other=0.0)
        ptr = n * stride_n + c * stride_c + offs
        x = tl.load(x_ptr + ptr, mask=mask, other=0.0)
        y = x * scale_c + shift_c
        tl.store(y_ptr + ptr, y, mask=mask)


class ModelNew(nn.Module):
    """BatchNorm2d optimized for contiguous float32 NCHW tensors on Ascend NPU."""

    def __init__(self, num_features: int = 64):
        super(ModelNew, self).__init__()
        self.bn = nn.BatchNorm2d(num_features=num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not getattr(x, "is_npu", False):
            raise ValueError("ModelNew expects an NPU tensor input")
        if x.dtype != torch.float32:
            raise TypeError("ModelNew only supports float32 inputs")
        if x.dim() != 4:
            raise ValueError("ModelNew expects NCHW 4D input")
        if not x.is_contiguous():
            raise ValueError("ModelNew expects contiguous NCHW inputs")

        bn = self.bn
        N, C, H, W = x.shape
        if C != bn.num_features:
            raise ValueError(
                "input channel count must match BatchNorm2d num_features")
        eps = bn.eps

        if bn.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = bn.momentum
        if bn.training and bn.track_running_stats:
            if bn.num_batches_tracked is not None:
                bn.num_batches_tracked.add_(1)
                if bn.momentum is None:
                    exponential_average_factor = 1.0 / float(
                        bn.num_batches_tracked.item())
            else:
                exponential_average_factor = 1.0

        use_batch_stats = bn.training or (not bn.track_running_stats)
        update_stats = bn.training and bn.track_running_stats
        stride_n, stride_c, stride_h, stride_w = x.stride()
        if stride_w != 1 or stride_h != W:
            raise ValueError("ModelNew expects contiguous NCHW layout")

        # 2048 keeps live fp32 vectors under UB while reducing partials 4x vs row-wise W=512.
        BLOCK_SIZE = 2048
        NUM_HW_BLOCKS = triton.cdiv(H * W, BLOCK_SIZE)
        NUM_PARTS = N * NUM_HW_BLOCKS
        BLOCK_PARTS = 1024
        NUM_PART_CHUNKS = triton.cdiv(NUM_PARTS, BLOCK_PARTS)

        scale = torch.empty(C, device=x.device, dtype=torch.float32)
        shift = torch.empty(C, device=x.device, dtype=torch.float32)
        if bn.track_running_stats and (bn.running_mean
                                       is not None) and (bn.running_var
                                                         is not None):
            running_mean = bn.running_mean
            running_var = bn.running_var
        else:
            running_mean = torch.zeros(C, device=x.device, dtype=torch.float32)
            running_var = torch.ones(C, device=x.device, dtype=torch.float32)

        affine_flag = int(bn.affine and (bn.weight is not None)
                          and (bn.bias is not None))
        if affine_flag:
            weight = bn.weight.to(device=x.device, dtype=torch.float32)
            bias = bn.bias.to(device=x.device, dtype=torch.float32)
        else:
            weight = torch.empty(1, device=x.device, dtype=torch.float32)
            bias = torch.empty(1, device=x.device, dtype=torch.float32)

        total_tiles = C * NUM_PARTS
        max_programs = 65535
        if use_batch_stats:
            partial_sum = torch.empty((C, NUM_PARTS),
                                      device=x.device,
                                      dtype=torch.float32)
            partial_sumsq = torch.empty((C, NUM_PARTS),
                                        device=x.device,
                                        dtype=torch.float32)
            if total_tiles > max_programs:
                n_programs = max_programs
                _bn_reduce_hw_block_persistent[(n_programs, )](
                    x,
                    partial_sum,
                    partial_sumsq,
                    n_programs,
                    total_tiles,
                    N,
                    H,
                    W,
                    stride_n,
                    stride_c,
                    NUM_HW_BLOCKS,
                    NUM_PARTS,
                    BLOCK_SIZE=BLOCK_SIZE,
                    num_warps=8,
                    num_stages=2,
                )
            else:
                _bn_reduce_hw_block[(C, NUM_PARTS)](
                    x,
                    partial_sum,
                    partial_sumsq,
                    N,
                    H,
                    W,
                    stride_n,
                    stride_c,
                    NUM_HW_BLOCKS,
                    BLOCK_SIZE=BLOCK_SIZE,
                    num_warps=8,
                    num_stages=2,
                )
        else:
            partial_sum = torch.empty(1, device=x.device, dtype=torch.float32)
            partial_sumsq = torch.empty(1,
                                        device=x.device,
                                        dtype=torch.float32)

        _bn_finalize_params_opt[(C, )](
            partial_sum,
            partial_sumsq,
            scale,
            shift,
            running_mean,
            running_var,
            weight,
            bias,
            NUM_PARTS,
            N * H * W,
            eps,
            exponential_average_factor,
            use_batch_stats,
            update_stats,
            affine_flag,
            BLOCK_PARTS=BLOCK_PARTS,
            NUM_PART_CHUNKS=NUM_PART_CHUNKS,
            num_warps=4,
            num_stages=2,
        )

        y = torch.empty_like(x)
        if total_tiles > max_programs:
            n_programs = max_programs
            _bn_apply_hw_block_persistent[(n_programs, )](
                x,
                y,
                scale,
                shift,
                n_programs,
                total_tiles,
                N,
                H,
                W,
                stride_n,
                stride_c,
                NUM_HW_BLOCKS,
                NUM_PARTS,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=8,
                num_stages=2,
            )
        else:
            _bn_apply_hw_block[(C, NUM_PARTS)](
                x,
                y,
                scale,
                shift,
                N,
                H,
                W,
                stride_n,
                stride_c,
                NUM_HW_BLOCKS,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=8,
                num_stages=2,
            )
        return y


batch_size = 64
features = 64
dim1 = 512
dim2 = 512


def get_inputs():
    x = torch.rand(batch_size, features, dim1, dim2)
    return [x]


def get_init_inputs():
    return [features]
