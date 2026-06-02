import triton
import triton.language as tl

@triton.jit
def _fused_linear_sigmoid_row_sum_kernel(
    x_ptr,         # *f32 / *f16 [B, K]
    w_ptr,         # *f32 / *f16 [H, K]
    b_ptr,         # *f32 / *f16 [H]
    out_ptr,       # *f32        [B]
    B, K, H,       # int32 sizes
    stride_xm, stride_xk,
    stride_wj, stride_wk,
    stride_b,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)  # one program per batch row
    if pid >= B:
        return

    # Pointer to row i in X
    x_row_ptr = x_ptr + pid * stride_xm

    # Accumulator for sum over hidden dim after sigmoid
    row_sum = tl.zeros((), dtype=tl.float32)

    j_arange = tl.arange(0, BLOCK_H)
    k_arange = tl.arange(0, BLOCK_K)

    # Fast path for K <= BLOCK_K: load x once and reuse for all hidden tiles
    if K <= BLOCK_K:
        k_offsets = k_arange
        k_mask = k_offsets < K
        x_vals = tl.load(
            x_row_ptr + k_offsets * stride_xk,
            mask=k_mask,
            other=0.0,
        ).to(tl.float32)  # [BLOCK_K]

        j_start = 0
        while j_start < H:
            j_offsets = j_start + j_arange
            j_mask = j_offsets < H

            # Compute linear outputs for this hidden tile
            w_tile = tl.load(
                w_ptr + j_offsets[:, None] * stride_wj + k_offsets[None, :] * stride_wk,
                mask=j_mask[:, None] & k_mask[None, :],
                other=0.0,
            ).to(tl.float32)  # [BLOCK_H, BLOCK_K]

            acc = tl.sum(w_tile * x_vals[None, :], axis=1)

            # Add bias
            b_vals = tl.load(b_ptr + j_offsets * stride_b, mask=j_mask, other=0.0).to(tl.float32)
            acc = acc + b_vals

            # Sigmoid and masked reduction over this hidden tile
            s = tl.sigmoid(acc)
            s = tl.where(j_mask, s, 0.0)
            row_sum += tl.sum(s, axis=0)

            j_start += BLOCK_H
    else:
        # General path for larger K
        j_start = 0
        while j_start < H:
            j_offsets = j_start + j_arange
            j_mask = j_offsets < H

            acc = tl.zeros((BLOCK_H,), dtype=tl.float32)

            # Loop over K dimension
            k_start = 0
            while k_start < K:
                k_offsets = k_start + k_arange
                k_mask = k_offsets < K

                # Load x[i, k] once per K-tile
                x_vals = tl.load(
                    x_row_ptr + k_offsets * stride_xk,
                    mask=k_mask,
                    other=0.0,
                ).to(tl.float32)  # [BLOCK_K]

                # Load W[j, k] tile
                w_tile = tl.load(
                    w_ptr + j_offsets[:, None] * stride_wj + k_offsets[None, :] * stride_wk,
                    mask=j_mask[:, None] & k_mask[None, :],
                    other=0.0,
                ).to(tl.float32)  # [BLOCK_H, BLOCK_K]

                # Fused multiply-accumulate along K
                acc += tl.sum(w_tile * x_vals[None, :], axis=1)
                k_start += BLOCK_K

            # Add bias
            b_vals = tl.load(b_ptr + j_offsets * stride_b, mask=j_mask, other=0.0).to(tl.float32)
            acc = acc + b_vals

            # Sigmoid and masked reduction over this hidden tile
            s = tl.sigmoid(acc)
            s = tl.where(j_mask, s, 0.0)
            row_sum += tl.sum(s, axis=0)

            j_start += BLOCK_H

    # Write per-row result
    tl.store(out_ptr + pid, row_sum)

@triton.jit
def _logsumexp_kernel(inp_ptr, out_ptr, B, BLOCK: tl.constexpr):
    # Optimized single-tile path when B <= BLOCK
    if B <= BLOCK:
        idx = tl.arange(0, BLOCK)
        mask = idx < B
        vals = tl.load(inp_ptr + idx, mask=mask, other=-1.0e30)
        m = tl.max(vals, axis=0)
        sum_exp = tl.sum(tl.exp(vals - m), axis=0)
        result = m + tl.log(sum_exp)
        tl.store(out_ptr, result)
        return

    # Fallback: numerically-stable tiled accumulation for arbitrary B
    acc_max = -1.0e30  # effectively -inf
    acc_sum = 0.0

    offset = 0
    while offset < B:
        idx = offset + tl.arange(0, BLOCK)
        mask = idx < B
        vals = tl.load(inp_ptr + idx, mask=mask, other=-1.0e30)

        tile_max = tl.max(vals, axis=0)
        tile_sum = tl.sum(tl.exp(vals - tile_max), axis=0)

        new_max = tl.maximum(acc_max, tile_max)
        acc_sum = acc_sum * tl.exp(acc_max - new_max) + tile_sum * tl.exp(tile_max - new_max)
        acc_max = new_max

        offset += BLOCK

    result = acc_max + tl.log(acc_sum)
    tl.store(out_ptr, result)

@triton.jit
def _fused_rowsum_logsumexp_kernel(
    x_ptr,         # *f32 [B, K]
    w_ptr,         # *f32 [H, K]
    b_ptr,         # *f32 [H]
    out_ptr,       # *f32 [1]
    B, K, H,
    stride_xm, stride_xk,
    stride_wj, stride_wk,
    stride_b,
    BLOCK_B: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Single-CTA persistent kernel: computes row-wise sum(sigmoid(xW^T+b)) and directly reduces with logsumexp over rows
    pid = tl.program_id(0)
    if pid != 0:
        return

    acc_max = -1.0e30  # running max for stable logsumexp
    acc_sum = 0.0      # running sum of exp shifted by acc_max

    rows_arange = tl.arange(0, BLOCK_B)
    j_arange = tl.arange(0, BLOCK_H)
    k_arange = tl.arange(0, BLOCK_K)

    b_start = 0
    while b_start < B:
        rows = b_start + rows_arange
        row_mask = rows < B

        # Accumulator for per-row sums after sigmoid over hidden dim
        row_sums = tl.zeros((BLOCK_B,), dtype=tl.float32)

        j_start = 0
        while j_start < H:
            j_offsets = j_start + j_arange
            j_mask = j_offsets < H

            # Accumulated GEMV for each row and hidden unit in this tile
            acc = tl.zeros((BLOCK_B, BLOCK_H), dtype=tl.float32)

            k_start = 0
            while k_start < K:
                k_offsets = k_start + k_arange
                k_mask = k_offsets < K

                # Load X tiles for multiple rows at once: [BLOCK_B, BLOCK_K]
                x_tile = tl.load(
                    x_ptr + rows[:, None] * stride_xm + k_offsets[None, :] * stride_xk,
                    mask=row_mask[:, None] & k_mask[None, :],
                    other=0.0,
                ).to(tl.float32)

                # Load W tile once and reuse across rows: [BLOCK_H, BLOCK_K]
                w_tile = tl.load(
                    w_ptr + j_offsets[:, None] * stride_wj + k_offsets[None, :] * stride_wk,
                    mask=j_mask[:, None] & k_mask[None, :],
                    other=0.0,
                ).to(tl.float32)

                # Accumulate: acc[r, j] += sum_k x_tile[r, k] * w_tile[j, k]
                acc += tl.sum(x_tile[:, None, :] * w_tile[None, :, :], axis=2)

                k_start += BLOCK_K

            # Add bias and apply sigmoid
            b_vals = tl.load(b_ptr + j_offsets * stride_b, mask=j_mask, other=0.0).to(tl.float32)
            acc = acc + b_vals[None, :]

            s = tl.sigmoid(acc)
            s = tl.where(row_mask[:, None] & j_mask[None, :], s, 0.0)

            # Reduce over hidden tile into per-row sums
            row_sums += tl.sum(s, axis=1)

            j_start += BLOCK_H

        # Tile-wise numerically stable accumulation into global logsumexp
        masked_vals = tl.where(row_mask, row_sums, -1.0e30)
        tile_max = tl.max(masked_vals, axis=0)
        tile_sum = tl.sum(tl.exp(masked_vals - tile_max), axis=0)

        new_max = tl.maximum(acc_max, tile_max)
        acc_sum = acc_sum * tl.exp(acc_max - new_max) + tile_sum * tl.exp(tile_max - new_max)
        acc_max = new_max

        b_start += BLOCK_B

    result = acc_max + tl.log(acc_sum)
    tl.store(out_ptr, result)
