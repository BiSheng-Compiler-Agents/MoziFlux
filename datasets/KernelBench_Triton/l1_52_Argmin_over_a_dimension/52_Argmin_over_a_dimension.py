import triton
import triton.language as tl

@triton.jit
def _argmin_row_kernel(
    x_ptr,
    out_ptr,
    rows,
    cols,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    if pid >= rows:
        return

    row_base = pid * cols
    best_val = tl.full((), float("inf"), dtype=tl.float32)
    best_idx = tl.zeros((), dtype=tl.int64)

    k0 = 0
    while k0 < cols:
        offsets = k0 + tl.arange(0, BLOCK_K)
        mask = offsets < cols
        values = tl.load(
            x_ptr + row_base + offsets,
            mask=mask,
            other=float("inf"),
            cache_modifier=".cg",
        ).to(tl.float32)

        tile_min = tl.min(values, axis=0)
        equal_mask = (values == tile_min) & mask
        invalid_index = tl.full((BLOCK_K,), cols, dtype=tl.int64)
        tile_indices = tl.where(equal_mask, offsets.to(tl.int64), invalid_index)
        tile_first_idx = tl.min(tile_indices, axis=0)

        should_update = (tile_min < best_val) | (
            (tile_min == best_val) & (tile_first_idx < best_idx)
        )
        best_val = tl.where(should_update, tile_min, best_val)
        best_idx = tl.where(should_update, tile_first_idx, best_idx)
        k0 += BLOCK_K

    tl.store(out_ptr + pid, best_idx)
