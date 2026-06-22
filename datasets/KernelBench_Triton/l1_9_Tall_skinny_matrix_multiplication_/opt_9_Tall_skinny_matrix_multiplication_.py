import os

import torch
import torch.nn as nn
import triton
import triton.language as tl

# Allow conditional global access in @jit'd functions
os.environ.setdefault("TRITON_ALLOW_NON_CONSTEXPR_GLOBALS", "1")

# --- Conditional CANN extension import ---
try:
    import triton.language.extra.cann.extension as al
    HAS_AL = True
except (ImportError, ModuleNotFoundError):
    HAS_AL = False


@triton.autotune(
    configs=[
        # Tall-skinny: large BLOCK_M reduces grid count
        triton.Config(
            {
                "BLOCK_M": 512,
                "BLOCK_N": 32,
                "BLOCK_K": 32,
                "GROUP_M": 4
            },
            num_warps=8,
            num_stages=4),
        triton.Config(
            {
                "BLOCK_M": 256,
                "BLOCK_N": 32,
                "BLOCK_K": 32,
                "GROUP_M": 4
            },
            num_warps=8,
            num_stages=4),
        triton.Config(
            {
                "BLOCK_M": 128,
                "BLOCK_N": 32,
                "BLOCK_K": 32,
                "GROUP_M": 8
            },
            num_warps=4,
            num_stages=4),
        triton.Config(
            {
                "BLOCK_M": 512,
                "BLOCK_N": 16,
                "BLOCK_K": 32,
                "GROUP_M": 4
            },
            num_warps=8,
            num_stages=4),
        triton.Config(
            {
                "BLOCK_M": 256,
                "BLOCK_N": 16,
                "BLOCK_K": 32,
                "GROUP_M": 4
            },
            num_warps=4,
            num_stages=4),
        triton.Config(
            {
                "BLOCK_M": 128,
                "BLOCK_N": 16,
                "BLOCK_K": 32,
                "GROUP_M": 8
            },
            num_warps=4,
            num_stages=4),
        triton.Config(
            {
                "BLOCK_M": 1024,
                "BLOCK_N": 16,
                "BLOCK_K": 32,
                "GROUP_M": 2
            },
            num_warps=8,
            num_stages=4),
        triton.Config(
            {
                "BLOCK_M": 64,
                "BLOCK_N": 32,
                "BLOCK_K": 32,
                "GROUP_M": 8
            },
            num_warps=4,
            num_stages=4),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _matmul_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    # 2D tile decomposition using a 1D grid with grouping along M dimension
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k_base = tl.arange(0, BLOCK_K)

    # --- Hoisted boundary masks (computed once outside K loop) ---
    row_mask = offs_m[:, None] < M  # BLOCK_M x 1
    col_mask = offs_n[None, :] < N  # 1 x BLOCK_N

    # --- Accumulator: start with zeros in fp32 ---
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    num_k_iters = tl.cdiv(K, BLOCK_K)

    for k_idx in tl.range(0, num_k_iters):
        k_offs = k_idx * BLOCK_K
        offs_k = offs_k_base + k_offs

        # Pointers for this K iteration (recomputed from base — no advancing pointers)
        A_block_ptrs = A_ptr + (offs_m[:, None] * stride_am +
                                offs_k[None, :] * stride_ak)
        B_block_ptrs = B_ptr + (offs_k[:, None] * stride_bk +
                                offs_n[None, :] * stride_bn)

        k_mask = offs_k < K
        a_mask = row_mask & k_mask[None, :]
        b_mask = k_mask[:, None] & col_mask

        # Use care_padding=False: safe for matmul (zero-input tiles contribute nothing to tl.dot)
        a = tl.load(A_block_ptrs, mask=a_mask, other=0.0, care_padding=False)
        b = tl.load(B_block_ptrs, mask=b_mask, other=0.0, care_padding=False)

        # Ascend-specific compile hint: only K dimension needs Cube padding
        if HAS_AL:
            al.compile_hint(a, "dot_pad_only_k")
            al.compile_hint(b, "dot_pad_only_k")

        # In-place accumulation: accumulates inside Cube hardware, eliminates 64 KB fp32 temp
        acc = tl.dot(a, b, acc)

    # --- Output: store with precomputed mask ---
    C_block_ptrs = C_ptr + (offs_m[:, None] * stride_cm +
                            offs_n[None, :] * stride_cn)
    tl.store(C_block_ptrs, acc, mask=row_mask & col_mask)


def _require_supported_runtime(tensor: torch.Tensor) -> None:
    if tensor.is_cuda or tensor.device.type == "npu":
        return
    if os.environ.get("TRITON_INTERPRET") == "1":
        return
    raise RuntimeError(
        "This operator requires CUDA or NPU tensors, or TRITON_INTERPRET=1 for Triton interpreter mode."
    )


def _validate_inputs(a: torch.Tensor,
                     b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("ModelNew expects two 2D tensors.")
    if a.shape[1] != b.shape[0]:
        raise ValueError(
            f"Incompatible shapes for matmul: {tuple(a.shape)} and {tuple(b.shape)}."
        )
    if a.device != b.device:
        raise ValueError("Inputs must be on the same device.")
    if a.dtype != b.dtype:
        raise ValueError("Inputs must have the same dtype.")
    if a.dtype not in {torch.float16, torch.bfloat16}:
        raise TypeError(
            f"Unsupported dtype for Triton tall-skinny matmul: {a.dtype}.")
    _require_supported_runtime(a)
    return a.contiguous(), b.contiguous()


def _triton_tall_skinny_matmul(a: torch.Tensor,
                               b: torch.Tensor) -> torch.Tensor:
    a, b = _validate_inputs(a, b)
    m, k = a.shape
    _, n = b.shape
    c_acc = torch.empty((m, n), device=a.device, dtype=torch.float32)

    # Conservative grid: use smallest BLOCK_M/N across autotune configs
    smallest_m = 64
    smallest_n = 16
    grid_m = triton.cdiv(m, smallest_m)
    grid_n = triton.cdiv(n, smallest_n)
    grid = (grid_m * grid_n, )

    _matmul_kernel[grid](
        a,
        b,
        c_acc,
        m,
        n,
        k,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c_acc.stride(0),
        c_acc.stride(1),
    )
    return c_acc.to(dtype=a.dtype)


class ModelNew(nn.Module):
    """
    Simple model that performs a single matrix multiplication (C = A * B)
    where one of the matrices is tall and skinny (M >> N or N >> M).
    """

    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs the matrix multiplication.

        Args:
            A (torch.Tensor): Input matrix of shape (M, K).
            B (torch.Tensor): Input matrix of shape (K, N).

        Returns:
            torch.Tensor: Output matrix of shape (M, N).
        """
        return _triton_tall_skinny_matmul(A, B)


# --- Benchmark support (kept for backward compatibility) ---
M = 16384 * 2
N = 16 * 2


def get_inputs():
    device = "npu" if hasattr(torch,
                              "npu") and torch.npu.is_available() else "cpu"
    A = torch.rand(M, N, device=device)
    B = torch.rand(N, M, device=device)
    return [A, B]


def get_init_inputs():
    return []
