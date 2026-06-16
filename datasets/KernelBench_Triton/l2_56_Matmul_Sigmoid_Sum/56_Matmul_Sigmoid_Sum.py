import triton
import triton.language as tl


@triton.jit
def _fused_linear_sigmoid_sum_kernel(
        x_ptr,  # float* [B, I]
        w_ptr,  # float* [H, I]
        b_ptr,  # float* [H]
        out_ptr,  # float* [B, 1]
        B: tl.constexpr,
        I: tl.constexpr,  # noqa: E741
        H: tl.constexpr,
        stride_xb,
        stride_xi,
        stride_wh,
        stride_wi,
        stride_bo,
        stride_ob,
        BLOCK_H: tl.constexpr,  # tile in hidden dimension (power-of-two)
        BLOCK_K: tl.constexpr,  # tile in input dimension (power-of-two)
):
    # One program per batch row
    pid_b = tl.program_id(0)

    # Accumulator for final sum over hidden dim
    acc_total = tl.zeros((), dtype=tl.float32)

    # Tile over hidden dimension
    h_start = 0
    while h_start < H:
        h_offsets = h_start + tl.arange(0, BLOCK_H)
        h_mask = h_offsets < H

        # Accumulator for current hidden tile (size BLOCK_H)
        z = tl.zeros((BLOCK_H, ), dtype=tl.float32)

        # Tile over input dimension
        k_start = 0
        while k_start < I:
            k_offsets = k_start + tl.arange(0, BLOCK_K)
            k_mask = k_offsets < I

            # Load a tile of x[b, k]
            x_ptrs = x_ptr + pid_b * stride_xb + k_offsets * stride_xi
            x_vals = tl.load(x_ptrs, mask=k_mask, other=0.0).to(tl.float32)

            # Load corresponding tile of W[h, k]
            w_ptrs = w_ptr + (h_offsets[:, None] * stride_wh +
                              k_offsets[None, :] * stride_wi)
            w_tile = tl.load(w_ptrs,
                             mask=h_mask[:, None] & k_mask[None, :],
                             other=0.0).to(tl.float32)

            # Accumulate z[h] += dot(W[h, k_tile], x[k_tile])
            z += tl.sum(w_tile * x_vals[None, :], axis=1)

            k_start += BLOCK_K

        # Add bias
        b_vals = tl.load(b_ptr + h_offsets * stride_bo, mask=h_mask,
                         other=0.0).to(tl.float32)
        z = z + b_vals

        # Sigmoid and reduce over current hidden tile
        s = 1.0 / (1.0 + tl.exp(-z))
        s = tl.where(h_mask, s, 0.0)
        acc_total += tl.sum(s, axis=0)

        h_start += BLOCK_H

    # Store result to out[b, 0]
    out_ptrs = out_ptr + pid_b * stride_ob
    tl.store(out_ptrs, acc_total)
