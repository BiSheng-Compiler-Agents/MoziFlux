import triton
import triton.language as tl

@triton.jit
def avgpool1d_forward_kernel(
    x_ptr,  # *[N_ROWS, L_IN]
    y_ptr,  # *[N_ROWS, L_OUT]
    N_ROWS: tl.constexpr,
    L_IN: tl.constexpr,
    L_OUT: tl.constexpr,
    N_COL_BLOCKS: tl.constexpr,
    stride_x_row: tl.constexpr,
    stride_y_row: tl.constexpr,
    STRIDE: tl.int32,
    PADDING: tl.int32,
    KERNEL_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row_id = tl.program_id(axis=0)
    if row_id >= N_ROWS:
        return

    # Row base pointers
    x_row_ptr = x_ptr + row_id * stride_x_row
    y_row_ptr = y_ptr + row_id * stride_y_row

    invK = 1.0 / float(KERNEL_SIZE)
    block_offsets = tl.arange(0, BLOCK)
    for col_block in tl.range(0, N_COL_BLOCKS):
        offs = col_block * BLOCK + block_offsets
        mask_o = offs < L_OUT
        j = offs * STRIDE - PADDING
        acc = tl.zeros([BLOCK], dtype=tl.float32)

        # Keep clamped addresses for safety while using L2-friendly cache modifier.
        for k in tl.static_range(0, KERNEL_SIZE):
            pos = j + k
            in_bounds = (pos >= 0) & (pos < L_IN)
            mask_k = in_bounds & mask_o
            pos_safe = tl.maximum(tl.minimum(pos, L_IN - 1), 0)
            vals_k = tl.load(
                x_row_ptr + pos_safe,
                mask=mask_k,
                other=0.0,
                cache_modifier=".cg",
            )
            acc += vals_k.to(tl.float32)

        out = acc * invK
        tl.store(y_row_ptr + offs, out, mask=mask_o)
