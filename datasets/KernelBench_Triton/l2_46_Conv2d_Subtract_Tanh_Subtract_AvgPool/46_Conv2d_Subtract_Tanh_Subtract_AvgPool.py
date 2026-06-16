import triton
import triton.language as tl


@triton.jit
def _fused_sub_tanh_sub_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    subtract1,
    subtract2,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # Load
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)

    # x - subtract1
    x = x - subtract1

    # Numerically-stable tanh using exponentials
    ax = tl.abs(x)
    e = tl.exp(-2.0 * ax)
    tanh_pos = (1.0 - e) / (1.0 + e)
    sign = tl.where(x >= 0.0, 1.0, -1.0)
    x = sign * tanh_pos

    # - subtract2 and store
    x = x - subtract2
    tl.store(y_ptr + offs, x, mask=mask)


@triton.jit
def _fused_tanh_avgpool_sub2_kernel(
        x_ptr,  # float* [N, C, H, W]
        y_ptr,  # float* [N, C, outH, outW]
        N,
        C,
        H,
        W,  # input dims
        outH,
        outW,  # output dims (after pooling)
        subtract1,  # float
        subtract2,  # float
        BLOCK_HW: tl.constexpr,  # number of output HW elements per program
        K: tl.
    constexpr,  # pooling kernel size; stride assumed = K (AvgPool2d default)
):
    # program ids
    pid_nc = tl.program_id(axis=0)  # over N*C
    pid_blk = tl.program_id(axis=1)  # over blocks of outH*outW

    # decode n, c
    n = pid_nc // C
    c = pid_nc % C

    # per-NC base strides
    in_nc_stride = H * W
    out_nc_stride = outH * outW

    # output linear indices this program handles
    offs_hw = pid_blk * BLOCK_HW + tl.arange(0, BLOCK_HW)
    mask_hw = offs_hw < (outH * outW)

    # map to (oh, ow)
    oh = offs_hw // outW
    ow = offs_hw % outW

    # starting input coordinates for each output (stride = K)
    ih0 = oh * K
    iw0 = ow * K

    # base pointers per (oh, ow) for the current (n, c)
    base_in = (n * C + c) * in_nc_stride + ih0 * W + iw0
    base_out = (n * C + c) * out_nc_stride + oh * outW + ow

    # accumulate tanh(x - subtract1) over KxK window
    acc = tl.zeros([BLOCK_HW], dtype=tl.float32)

    # Unrolled pooling window
    for rr in range(0, K):
        row_off = base_in + rr * W
        for ss in range(0, K):
            idx = row_off + ss
            x = tl.load(x_ptr + idx, mask=mask_hw, other=0.0)
            # stable tanh: tanh(x) = sign(x)*(1 - e)/(1 + e) where e=exp(-2*|x|)
            x = x - subtract1
            ax = tl.abs(x)
            e = tl.exp(-2.0 * ax)
            t = (1.0 - e) / (1.0 + e)
            x = tl.where(x >= 0.0, t, -t)
            acc += x

    inv_area = 1.0 / (K * K)
    out_val = acc * inv_area - subtract2
    tl.store(y_ptr + base_out, out_val, mask=mask_hw)
