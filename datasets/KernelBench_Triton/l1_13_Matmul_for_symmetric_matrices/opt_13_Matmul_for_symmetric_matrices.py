"""
Optimized symmetric matrix multiplication kernel for Ascend NPU.
Exploits: larger block sizes for Cube utilization, GROUP_M swizzle for L2 reuse,
tl.static_range (via K: tl.constexpr) for K-loop unrolling,
al.multibuffer for DMA/Cube overlap, al.compile_hint for reduced UB padding,
mask hoisting, care_padding=False, and diagonal scheduling for large matrices.

Baseline: BLOCK_M=32, BLOCK_N=32, BLOCK_K=32, 2D grid, dynamic K loop,
per-iteration masks, no compile_hint, no multibuffer.
CUBE util: 5.1%, SCALARLDST: 48602 cyc (ST_XD_XN_IMM 81%), 512 RV_VLDI+512 RV_VSTI
"""
import torch
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al

# ── Autotune Configurations ──────────────────────────────────────────────────


@triton.autotune(
    configs=[
        # Large tiles for best Cube utilization on medium-large matrices
        triton.Config({
            "BLOCK_M": 128,
            "BLOCK_N": 128,
            "BLOCK_K": 64
        },
                      num_stages=0,
                      num_warps=8),
        triton.Config({
            "BLOCK_M": 128,
            "BLOCK_N": 128,
            "BLOCK_K": 32
        },
                      num_stages=0,
                      num_warps=8),
        triton.Config({
            "BLOCK_M": 128,
            "BLOCK_N": 64,
            "BLOCK_K": 64
        },
                      num_stages=0,
                      num_warps=8),
        triton.Config({
            "BLOCK_M": 128,
            "BLOCK_N": 64,
            "BLOCK_K": 32
        },
                      num_stages=0,
                      num_warps=8),
        triton.Config({
            "BLOCK_M": 64,
            "BLOCK_N": 128,
            "BLOCK_K": 64
        },
                      num_stages=0,
                      num_warps=8),
        triton.Config({
            "BLOCK_M": 64,
            "BLOCK_N": 64,
            "BLOCK_K": 64
        },
                      num_stages=0,
                      num_warps=8),
        triton.Config({
            "BLOCK_M": 256,
            "BLOCK_N": 64,
            "BLOCK_K": 32
        },
                      num_stages=0,
                      num_warps=8),
        triton.Config({
            "BLOCK_M": 64,
            "BLOCK_N": 256,
            "BLOCK_K": 32
        },
                      num_stages=0,
                      num_warps=8),
        # Fallback for very small matrices
        triton.Config({
            "BLOCK_M": 32,
            "BLOCK_N": 32,
            "BLOCK_K": 128
        },
                      num_stages=0,
                      num_warps=8),
    ],
    key=["M", "N", "K"],
)
# ── Optimized Kernel ─────────────────────────────────────────────────────────

@triton.jit
def _symmetric_matmul_kernel_opt(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K: tl.constexpr,
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
    DIAG_THRESHOLD: tl.constexpr,
):
    # ── Number of blocks (constexpr because K is constexpr) ──
    NUM_BLOCKS_M = tl.cdiv(M, BLOCK_M)
    NUM_BLOCKS_N = tl.cdiv(N, BLOCK_N)
    tl.cdiv(K, BLOCK_K)  # compile-time when K is constexpr

    pid = tl.program_id(0)

    # ── Diagonal scheduling for large matrices ──
    if NUM_BLOCKS_M >= DIAG_THRESHOLD and NUM_BLOCKS_N >= DIAG_THRESHOLD:
        NUM_TILES = NUM_BLOCKS_M * NUM_BLOCKS_N
        for block_idx in range(pid, NUM_TILES, tl.num_programs(0)):
            task_m = block_idx % NUM_BLOCKS_M
            task_n = block_idx // NUM_BLOCKS_M
            _compute_tile(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak,
                          stride_bk, stride_bn, stride_cm, stride_cn, BLOCK_M,
                          BLOCK_N, BLOCK_K, task_m, task_n)
    else:
        # ── GROUP_M swizzle (1D → 2D mapping) ──
        num_pid_m = NUM_BLOCKS_M
        num_pid_n = NUM_BLOCKS_N
        num_pid_in_group = GROUP_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = min(GROUP_M, num_pid_m - first_pid_m)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        if pid_m < NUM_BLOCKS_M and pid_n < NUM_BLOCKS_N:
            _compute_tile(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak,
                          stride_bk, stride_bn, stride_cm, stride_cn, BLOCK_M,
                          BLOCK_N, BLOCK_K, pid_m, pid_n)


@triton.jit
def _compute_tile(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K: tl.constexpr,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    pid_m,
    pid_n,
):
    # ── NUM_BLOCKS_K is constexpr because K and BLOCK_K are constexpr ──
    NUM_BLOCKS_K = tl.cdiv(K, BLOCK_K)

    # ── Row/column offsets (hoisted) ──
    rm = pid_m * BLOCK_M
    rn = pid_n * BLOCK_N
    offs_m = rm + tl.arange(0, BLOCK_M)
    offs_n = rn + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # ── Boundary masks (hoisted — computed once) ──
    m_mask = offs_m < M
    n_mask = offs_n < N

    # ── Accumulator (FP32) ──
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # ── K loop — regular range (NUM_BLOCKS_K is constexpr for compile-time scheduling) ──
    for k_idx in range(0, NUM_BLOCKS_K):
        k_start = k_idx * BLOCK_K

        a_ptrs = a_ptr + offs_m[:, None] * stride_am + (
            k_start + offs_k)[None, :] * stride_ak
        b_ptrs = b_ptr + (k_start + offs_k
                          )[:, None] * stride_bk + offs_n[None, :] * stride_bn

        k_mask_a = (k_start + offs_k)[None, :] < K
        a_mask = m_mask[:, None] & k_mask_a
        k_mask_b = (k_start + offs_k)[:, None] < K
        b_mask = k_mask_b & n_mask[None, :]

        a = tl.load(a_ptrs, mask=a_mask, other=0.0, care_padding=False)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0, care_padding=False)

        al.compile_hint(a, "dot_pad_only_k")
        al.compile_hint(b, "dot_pad_only_k")

        al.multibuffer(a, size=2)
        al.multibuffer(b, size=2)

        acc += tl.dot(a, b)

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(c_ptrs, acc.to(tl.float32), mask=c_mask)


# ── Host dispatch ────────────────────────────────────────────────────────────


def symmetric_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Compute C = A @ B where A, B are symmetric FP32 matrices (row-major).
    Uses GROUP_M swizzle and diagonal scheduling for efficiency.
    """
    assert a.dim() == 2 and b.dim() == 2, "Only 2D matmul supported"
    M, K_a = a.shape
    K_b, N = b.shape
    assert K_a == K_b, f"Inner dimensions must match: {K_a} vs {K_b}"
    K = K_a

    c = torch.empty((M, N), device=a.device, dtype=torch.float32)

    def grid_fn(META):
        BLOCK_M = META["BLOCK_M"]
        BLOCK_N = META["BLOCK_N"]
        num_m = triton.cdiv(M, BLOCK_M)
        num_n = triton.cdiv(N, BLOCK_N)
        # GROUP_M swizzle mapping: num_pid_in_group = GROUP_M * num_n
        # Total programs limited to physical cores (96 on Ascend950)
        group_m = META.get("GROUP_M", 8)
        group_m * num_n
        total = num_m * num_n
        return (min(total, 96), )

    _symmetric_matmul_kernel_opt[grid_fn](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        GROUP_M=8,
        DIAG_THRESHOLD=6,
    )
    return c


# ── nn.Module wrapper ────────────────────────────────────────────────────────


class ModelNew(torch.nn.Module):
    """Host interface for the optimized symmetric matmul kernel."""

    def __init__(self, M: int, N: int, K: int):
        super().__init__()
        self.M = M
        self.N = N
        self.K = K

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return symmetric_matmul(a, b)


# ── Utility functions ────────────────────────────────────────────────────────


def get_init_inputs():
    """Return default init args for ModelNew."""
    return (512, 512, 512)


def get_inputs():
    """Return example input tensors for ModelNew.forward()."""
    M, N, K = 512, 512, 512
    a = torch.randn(M, K, device="npu", dtype=torch.float32)
    b = torch.randn(K, N, device="npu", dtype=torch.float32)
    # Symmetrize
    a = a + a.T
    b = b + b.T
    return (a, b)
