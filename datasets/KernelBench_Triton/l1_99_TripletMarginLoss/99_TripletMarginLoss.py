import triton
import triton.language as tl

@triton.jit
def _triplet_margin_row_kernel(
    anchor_ptr, pos_ptr, neg_ptr, out_ptr,
    B, D,
    stride_a0, stride_a1,
    stride_p0, stride_p1,
    stride_n0, stride_n1,
    eps, margin,
    BLOCK_SIZE: tl.constexpr,
    N_ITERS: tl.constexpr,
):
    row = tl.program_id(0)
    valid_row = row < B

    a_row_ptr = anchor_ptr + row * stride_a0
    p_row_ptr = pos_ptr + row * stride_p0
    n_row_ptr = neg_ptr + row * stride_n0

    acc_ap = tl.zeros((), dtype=tl.float32)
    acc_an = tl.zeros((), dtype=tl.float32)

    for i in tl.static_range(N_ITERS):
        offs = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = valid_row & (offs < D)

        a = tl.load(a_row_ptr + offs * stride_a1, mask=mask, other=0.0)
        p = tl.load(p_row_ptr + offs * stride_p1, mask=mask, other=0.0)
        n = tl.load(n_row_ptr + offs * stride_n1, mask=mask, other=0.0)

        da = a - p
        dn = a - n

        acc_ap += tl.sum(da * da, axis=0)
        acc_an += tl.sum(dn * dn, axis=0)

    d_ap = tl.sqrt(acc_ap + eps)
    d_an = tl.sqrt(acc_an + eps)
    loss = tl.maximum(d_ap - d_an + margin, 0.0)

    tl.store(out_ptr + row, loss, mask=valid_row)
