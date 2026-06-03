import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
except Exception as exc:
    triton = None
    tl = None
    _TRITON_IMPORT_ERROR = exc
else:
    _TRITON_IMPORT_ERROR = None


def _is_npu_tensor(x: torch.Tensor) -> bool:
    return bool(getattr(x, "is_npu", False) or x.device.type == "npu")


if triton is not None:
    @triton.jit
    def _mse_partial_sum_kernel(
        x_ptr,  # *T
        y_ptr,  # *T
        out_ptr,  # *fp32, single element (accumulator)
        n_elements,  # total number of elements
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        num_pid = tl.num_programs(axis=0)

        acc = 0.0  # scalar fp32 accumulator
        arange = tl.arange(0, BLOCK_SIZE)
        tl.multiple_of(arange, 128)

        # Each program processes strided chunks; we unroll by 2 to reduce loop overhead
        offset = pid * BLOCK_SIZE
        stride = BLOCK_SIZE * num_pid

        while offset < n_elements:
            offsets0 = offset + arange
            tl.max_contiguous(offsets0, BLOCK_SIZE)
            mask0 = offsets0 < n_elements
            x0 = tl.load(x_ptr + offsets0, mask=mask0, other=0.0)
            y0 = tl.load(y_ptr + offsets0, mask=mask0, other=0.0)
            diff0 = (x0 - y0).to(tl.float32)
            acc += tl.sum(diff0 * diff0, axis=0)

            offsets1 = offset + stride + arange
            tl.max_contiguous(offsets1, BLOCK_SIZE)
            mask1 = offsets1 < n_elements
            x1 = tl.load(x_ptr + offsets1, mask=mask1, other=0.0)
            y1 = tl.load(y_ptr + offsets1, mask=mask1, other=0.0)
            diff1 = (x1 - y1).to(tl.float32)
            acc += tl.sum(diff1 * diff1, axis=0)

            offset += 2 * stride

        tl.atomic_add(out_ptr + 0, acc)
else:
    _mse_partial_sum_kernel = None


class ModelNew(nn.Module):
    """
    A model that computes the Mean Squared Error loss for regression tasks.

    Parameters:
        None
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        # Lazy-initialized accumulator buffer to reduce per-call allocations
        self._acc_buf = None
        self._acc_buf_device = None

    def forward(self, predictions, targets):
        if predictions.shape != targets.shape:
            raise ValueError("predictions and targets must have the same shape")
        if predictions.device != targets.device:
            raise ValueError("predictions and targets must be on the same device")
        if not _is_npu_tensor(predictions) or not _is_npu_tensor(targets):
            raise RuntimeError("ModelNew expects Ascend NPU tensors")
        if _TRITON_IMPORT_ERROR is not None:
            raise RuntimeError("Triton is required for ModelNew on Ascend NPU") from _TRITON_IMPORT_ERROR
        if predictions.dtype not in (torch.float16, torch.float32, torch.bfloat16):
            raise TypeError(f"unsupported dtype for predictions: {predictions.dtype}")
        if targets.dtype not in (torch.float16, torch.float32, torch.bfloat16):
            raise TypeError(f"unsupported dtype for targets: {targets.dtype}")

        x = predictions
        y = targets
        if not x.is_contiguous():
            x = x.contiguous()
        if not y.is_contiguous():
            y = y.contiguous()
        x = x.view(-1)
        y = y.view(-1)
        n = x.numel()
        if n == 0:
            raise ValueError("predictions and targets must be non-empty")

        dev = x.device
        if (self._acc_buf is None) or (self._acc_buf_device != dev) or (self._acc_buf.dtype != torch.float32):
            self._acc_buf = torch.zeros(1, device=dev, dtype=torch.float32)
            self._acc_buf_device = dev
        else:
            self._acc_buf.zero_()

        BLOCK_SIZE = 4096

        grid = lambda META: (triton.cdiv(n, META['BLOCK_SIZE']),)
        _mse_partial_sum_kernel[grid](
            x, y, self._acc_buf, n,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4,   # tuned lower to reduce scheduling overhead for this bandwidth-bound op
            num_stages=2,  # lightweight pipelining
        )

        mean = self._acc_buf[0] / n
        # Match PyTorch dtype promotion for (pred - target)
        out_dtype = torch.result_type(predictions, targets)
        return mean.to(out_dtype)
batch_size = 32768
input_shape = (32768,)
dim = 1

def get_inputs():
    scale = torch.rand(())
    return [torch.rand(batch_size, *input_shape)*scale, torch.rand(batch_size, *input_shape)]
def get_init_inputs():
    return []
