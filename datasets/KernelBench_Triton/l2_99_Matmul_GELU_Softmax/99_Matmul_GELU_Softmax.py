import os
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _linear_gelu_softmax_rowwise(
    x_ptr,  # [B, K]
    w_ptr,  # [N, K]
    b_ptr,  # [N]
    y_ptr,  # [B, N]
    stride_x,  # stride between rows of x (in elements)
    stride_w_n,  # stride for weight along N (in elements)
    stride_w_k,  # stride for weight along K (in elements)
    stride_y,  # stride between rows of y (in elements)
    B,
    K,
    N,  # dimensions
    NUM_N_TILES: tl.constexpr,
    NUM_K_TILES: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    x_row_ptr = x_ptr + pid * stride_x
    y_row_ptr = y_ptr + pid * stride_y

    cols_n = tl.arange(0, BLOCK_N)
    cols_k = tl.arange(0, BLOCK_K)

    inv_sqrt2 = 0.7071067811865476
    neg_inf = -float("inf")

    # Fast path when the entire N fits in one tile: keep everything in registers
    if NUM_N_TILES == 1:
        j = cols_n
        j_mask = j < N

        acc = tl.zeros([BLOCK_N], dtype=tl.float32)
        k_start = 0
        for _ in tl.static_range(NUM_K_TILES):
            k = k_start + cols_k
            k_mask = k < K

            x_vals = tl.load(x_row_ptr + k, mask=k_mask,
                             other=0.0).to(tl.float32)

            w_ptrs = w_ptr + j[:, None] * stride_w_n + k[None, :] * stride_w_k
            wk_mask = j_mask[:, None] & k_mask[None, :]
            w_vals = tl.load(w_ptrs,
                             mask=wk_mask,
                             other=0.0,
                             cache_modifier=".cg").to(tl.float32)

            acc += tl.sum(w_vals * x_vals[None, :], axis=1)
            k_start += BLOCK_K

        bias_vals = tl.load(b_ptr + j, mask=j_mask, other=0.0).to(tl.float32)
        logits = acc + bias_vals
        gelu_vals = 0.5 * logits * (1.0 + tl.erf(logits * inv_sqrt2))

        gelu_masked = tl.where(j_mask, gelu_vals, neg_inf)
        row_max = tl.max(gelu_masked, axis=0)
        z = gelu_vals - row_max
        num = tl.exp(z)
        num = tl.where(j_mask, num, 0.0)
        denom = tl.sum(num, axis=0)
        out = num / denom
        tl.store(y_row_ptr + j, out, mask=j_mask)
        return

    # Generic path for multi-tile N: 3-pass algorithm with minimal recomputation
    row_max = neg_inf
    n_start = 0
    for _ in tl.static_range(NUM_N_TILES):
        j = n_start + cols_n
        j_mask = j < N

        acc = tl.zeros([BLOCK_N], dtype=tl.float32)
        k_start = 0
        for __ in tl.static_range(NUM_K_TILES):
            k = k_start + cols_k
            k_mask = k < K

            x_vals = tl.load(x_row_ptr + k, mask=k_mask,
                             other=0.0).to(tl.float32)

            w_ptrs = w_ptr + j[:, None] * stride_w_n + k[None, :] * stride_w_k
            wk_mask = j_mask[:, None] & k_mask[None, :]
            w_vals = tl.load(w_ptrs,
                             mask=wk_mask,
                             other=0.0,
                             cache_modifier=".cg").to(tl.float32)

            acc += tl.sum(w_vals * x_vals[None, :], axis=1)
            k_start += BLOCK_K

        bias_vals = tl.load(b_ptr + j, mask=j_mask, other=0.0).to(tl.float32)
        logits = acc + bias_vals
        gelu_vals = 0.5 * logits * (1.0 + tl.erf(logits * inv_sqrt2))

        tl.store(y_row_ptr + j, gelu_vals, mask=j_mask)
        row_max = tl.maximum(
            row_max, tl.max(tl.where(j_mask, gelu_vals, neg_inf), axis=0))
        n_start += BLOCK_N

    denom = 0.0
    n_start = 0
    for _ in tl.static_range(NUM_N_TILES):
        j = n_start + cols_n
        j_mask = j < N
        gelu_vals = tl.load(y_row_ptr + j, mask=j_mask, other=neg_inf)
        num = tl.exp(gelu_vals - row_max)
        tl.store(y_row_ptr + j, tl.where(j_mask, num, 0.0), mask=j_mask)
        denom += tl.sum(tl.where(j_mask, num, 0.0), axis=0)
        n_start += BLOCK_N

    inv_denom = 1.0 / denom
    n_start = 0
    for _ in tl.static_range(NUM_N_TILES):
        j = n_start + cols_n
        j_mask = j < N
        numer = tl.load(y_row_ptr + j, mask=j_mask, other=0.0)
        out = numer * inv_denom
        tl.store(y_row_ptr + j, out, mask=j_mask)
        n_start += BLOCK_N


def _next_power_of_two(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << ((n - 1).bit_length())


def _require_supported_runtime(tensor: torch.Tensor) -> None:
    if tensor.is_cuda or tensor.device.type == "npu":
        return
    if os.environ.get("TRITON_INTERPRET") == "1":
        return
    raise RuntimeError(
        "This operator requires CUDA or NPU tensors, or TRITON_INTERPRET=1.")


def _validate_inputs(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if x.ndim != 2 or weight.ndim != 2:
        raise ValueError("Expected x and weight to be 2D tensors.")
    if x.shape[1] != weight.shape[1]:
        raise ValueError(
            f"Incompatible shapes for fused linear: x={tuple(x.shape)}, weight={tuple(weight.shape)}."
        )
    if x.device != weight.device:
        raise ValueError("x and weight must be on the same device.")
    if bias is not None:
        if bias.ndim != 1 or bias.shape[0] != weight.shape[0]:
            raise ValueError(
                "bias must be a 1D tensor with shape [out_features].")
        if bias.device != x.device:
            raise ValueError("bias must be on the same device as x.")
    if x.dtype != weight.dtype or (bias is not None and bias.dtype != x.dtype):
        raise ValueError("x, weight, and bias must share the same dtype.")
    if x.dtype not in {torch.float16, torch.float32}:
        raise TypeError(f"Unsupported dtype for fused operator: {x.dtype}.")
    _require_supported_runtime(x)

    x = x.contiguous()
    weight = weight.contiguous()
    if bias is None:
        bias = torch.zeros(weight.shape[0],
                           device=weight.device,
                           dtype=weight.dtype)
    else:
        bias = bias.contiguous()
    return x, weight, bias


def matmul_gelu_softmax(x: torch.Tensor,
                        weight: torch.Tensor,
                        bias: torch.Tensor | None = None) -> torch.Tensor:
    x, weight, bias = _validate_inputs(x, weight, bias)
    batch_size, in_features = x.shape
    out_features = weight.shape[0]
    output = torch.empty((batch_size, out_features),
                         device=x.device,
                         dtype=x.dtype)

    block_n = min(128, max(16, _next_power_of_two(out_features)))
    block_k = min(128, max(32, _next_power_of_two(in_features)))
    num_n_tiles = triton.cdiv(out_features, block_n)
    num_k_tiles = triton.cdiv(in_features, block_k)
    num_warps = 1 if block_n <= 32 else 2

    _linear_gelu_softmax_rowwise[(batch_size, )](
        x,
        weight,
        bias,
        output,
        x.stride(0),
        weight.stride(0),
        weight.stride(1),
        output.stride(0),
        batch_size,
        in_features,
        out_features,
        NUM_N_TILES=num_n_tiles,
        NUM_K_TILES=num_k_tiles,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=num_warps,
        num_stages=2,
    )
    return output


DEFAULT_BATCH_SIZE = 1024
DEFAULT_IN_FEATURES = 8192
DEFAULT_OUT_FEATURES = 8192


class ModelNew(nn.Module):
    """
    Simple model that performs a matrix multiplication, applies GELU, and then applies Softmax.
    """

    def __init__(self, in_features=None, out_features=None):
        super(ModelNew, self).__init__()
        if in_features is None:
            in_features = DEFAULT_IN_FEATURES
        if out_features is None:
            out_features = DEFAULT_OUT_FEATURES
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return matmul_gelu_softmax(x, self.linear.weight, self.linear.bias)


batch_size = 1024
in_features = 8192
out_features = 8192


def get_inputs():
    device = "npu" if hasattr(torch,
                              "npu") and torch.npu.is_available() else "cpu"
    return [torch.rand(batch_size, in_features, device=device)]


def get_init_inputs():
    return [in_features, out_features]
