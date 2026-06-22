import torch
import torch.nn as nn
import triton
import triton.language as tl


def _is_npu_tensor(x: torch.Tensor) -> bool:
    return bool(getattr(x, "is_npu", False))


@triton.jit
def _prod_dim1_block_cumprod_kernel(
    x_ptr,
    y_ptr,
    B,
    M,
    K,
    stride_b,
    stride_m,
    stride_k,
    stride_ob,
    stride_ok,
    total_tiles,
    n_k_tiles,
    n_programs,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = tl.arange(0, BLOCK_M)
    offs_k_base = tl.max_contiguous(tl.multiple_of(tl.arange(0, BLOCK_K), 16),
                                    BLOCK_K)

    tile = pid
    while tile < total_tiles:
        b = tile // n_k_tiles
        k_blk = tile - b * n_k_tiles
        offs_k = k_blk * BLOCK_K + offs_k_base
        mask_k = offs_k < K
        acc = tl.full([BLOCK_K], 1.0, dtype=tl.float32)

        for m0 in tl.range(0, M, BLOCK_M):
            m_idxs = m0 + offs_m
            vals = tl.load(
                x_ptr + b * stride_b + m_idxs[:, None] * stride_m +
                offs_k[None, :] * stride_k,
                mask=(m_idxs[:, None] < M) & mask_k[None, :],
                other=1.0,
                eviction_policy="evict_first",
            ).to(tl.float32)
            scan = tl.cumprod(vals, axis=0)
            last_rel = tl.minimum(BLOCK_M, M - m0) - 1
            block_prod = tl.sum(tl.where(offs_m[:, None] == last_rel, scan,
                                         0.0),
                                axis=0)
            acc *= block_prod

        tl.store(y_ptr + b * stride_ob + offs_k * stride_ok, acc, mask=mask_k)
        tile += n_programs


class ModelNew(nn.Module):
    """Product reduction over dim=1 for a 3D tensor."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return product_reduction_over_a_dimension(x, self.dim)


def product_reduction_over_a_dimension(x: torch.Tensor,
                                       dim: int) -> torch.Tensor:
    if not _is_npu_tensor(x):
        raise RuntimeError(
            "product_reduction_over_a_dimension expects an Ascend NPU tensor")
    if x.dim() != 3:
        raise ValueError(
            f"product_reduction_over_a_dimension expects a 3D tensor, got {x.dim()}D"
        )
    if x.dtype not in (torch.float16, torch.float32, torch.bfloat16):
        raise TypeError(f"unsupported dtype for product reduction: {x.dtype}")

    dim = dim if dim >= 0 else x.dim() + dim
    if dim != 1:
        raise NotImplementedError(
            f"product_reduction_over_a_dimension currently supports only dim=1/-2, got dim={dim}"
        )

    x_contig = x.contiguous()
    B, M, K = x_contig.shape
    y = torch.empty((B, K), device=x_contig.device, dtype=x_contig.dtype)
    sB, sM, sK = x_contig.stride()
    oB, oK = y.stride()

    block_m = 64
    block_k = 64
    n_k_tiles = triton.cdiv(K, block_k)
    total_tiles = B * n_k_tiles
    n_programs = min(total_tiles, 65535)
    _prod_dim1_block_cumprod_kernel[(n_programs, )](
        x_contig,
        y,
        B,
        M,
        K,
        sB,
        sM,
        sK,
        oB,
        oK,
        total_tiles,
        n_k_tiles,
        n_programs,
        BLOCK_M=block_m,
        BLOCK_K=block_k,
        num_warps=4,
        num_stages=2,
    )
    return y


batch_size = 16
dim1 = 256
dim2 = 256
reduction_dim = 1


def get_inputs():
    device = "npu" if hasattr(torch,
                              "npu") and torch.npu.is_available() else "cpu"
    x = torch.randn(batch_size, dim1, dim2, device=device)
    return [x]


def get_init_inputs():
    return [reduction_dim]
