import torch
import torch.nn as nn
import triton
import triton.language as tl
import triton.runtime.driver as driver


@triton.jit
def _gemv_vector_opt_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    M: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    tile_count_m = tl.cdiv(M, BLOCK_M)

    offs_m = tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)

    for tile_m in tl.range(pid, tile_count_m, nprog):
        rows = tile_m * BLOCK_M + offs_m
        mask_m = rows < M
        row_offsets = rows[:, None] * stride_am
        acc = tl.zeros((BLOCK_M, ), dtype=tl.float32)

        for k0 in tl.range(0, K, BLOCK_K):
            cols = k0 + offs_k
            mask_k = cols < K
            a_ptrs = A_ptr + row_offsets + cols[None, :] * stride_ak
            b_ptrs = B_ptr + cols * stride_bk
            a_tile = tl.load(
                a_ptrs,
                mask=mask_m[:, None] & mask_k[None, :],
                other=0.0,
                care_padding=False,
            )
            b_tile = tl.load(
                b_ptrs,
                mask=mask_k,
                other=0.0,
                care_padding=False,
            )
            acc += tl.sum(a_tile.to(tl.float32) *
                          b_tile.to(tl.float32)[None, :],
                          axis=1)

        tl.store(C_ptr + rows * stride_cm, acc, mask=mask_m)


class ModelNew(nn.Module):
    """Matrix-vector multiplication C = A @ B using a widened reduction tile."""

    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        if A.ndim != 2 or B.ndim != 2:
            raise ValueError("ModelNew expects 2D inputs A and B.")
        if B.shape[1] != 1:
            raise ValueError("ModelNew expects B to have shape (K, 1).")
        if A.shape[1] != B.shape[0]:
            raise ValueError("ModelNew requires A.shape[1] == B.shape[0].")
        if A.device.type != "npu" or B.device.type != "npu":
            raise ValueError(
                "ModelNew requires both inputs to be on Ascend NPU.")
        if A.dtype != B.dtype:
            raise ValueError(
                "ModelNew requires A and B to have the same dtype.")
        if A.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError(
                "ModelNew supports float16, bfloat16, and float32 inputs.")

        M, K = A.shape
        A_ctg = A.contiguous()
        B_ctg = B.contiguous()
        C = torch.empty((M, 1), device=A.device, dtype=A.dtype)

        BLOCK_M = 64
        BLOCK_K = 512
        tile_count_m = triton.cdiv(M, BLOCK_M)
        device = torch.npu.current_device()
        props = driver.active.utils.get_device_properties(device)
        num_vectorcore = props.get("num_vectorcore", tile_count_m)
        grid = (max(1, min(tile_count_m, num_vectorcore)), )

        _gemv_vector_opt_kernel[grid](
            A_ctg,
            B_ctg,
            C,
            A_ctg.stride(0),
            A_ctg.stride(1),
            B_ctg.stride(0),
            B_ctg.stride(1),
            C.stride(0),
            C.stride(1),
            M=M,
            K=K,
            BLOCK_M=BLOCK_M,
            BLOCK_K=BLOCK_K,
        )
        return C


M = 256 * 8  # 2048
K = 131072 * 8  # 1048576


def get_inputs():
    A = torch.rand(M, K)
    B = torch.rand(K, 1)
    return [A, B]


def get_init_inputs():
    return []
