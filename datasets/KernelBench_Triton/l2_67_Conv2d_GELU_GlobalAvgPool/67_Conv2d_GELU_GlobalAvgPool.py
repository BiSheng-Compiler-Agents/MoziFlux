import triton
import triton.language as tl

@triton.jit
def _gelu_gap2d_fused_row_kernel(
    x_ptr,                      # *f32/ *f16 input tensor pointer [N, C, H, W]
    y_ptr,                      # *f32 output tensor pointer [N, C]
    C, H, W,                    # ints
    stride_n, stride_c, stride_h, stride_w,  # strides for x in elements
    out_stride_n, out_stride_c,              # strides for y in elements
    BLOCK_W: tl.constexpr,                   # tile size across flattened H*W
):
    pid = tl.program_id(axis=0)
    n = pid // C
    c = pid % C

    # base pointer for this (n, c) plane
    base = n * stride_n + c * stride_c

    # Flatten spatial dims; host makes x contiguous so H*W is contiguous
    total_hw = H * W
    idx = tl.arange(0, BLOCK_W)

    # Vector accumulator to minimize per-iteration reductions
    acc_vec = tl.zeros((BLOCK_W,), dtype=tl.float32)

    inv_sqrt2 = 0.7071067811865476  # 1/sqrt(2)

    # Tile over flattened plane
    for start in range(0, total_hw, BLOCK_W):
        offs = base + start + idx
        mask = (start + idx) < total_hw
        vals = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        # GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
        gelu_vals = 0.5 * vals * (1.0 + tl.erf(vals * inv_sqrt2))
        gelu_vals = tl.where(mask, gelu_vals, 0.0)
        acc_vec += gelu_vals

    acc = tl.sum(acc_vec, axis=0)
    mean_val = acc / tl.full((), total_hw, dtype=tl.float32)

    out_off = n * out_stride_n + c * out_stride_c
    tl.store(y_ptr + out_off, mean_val)
