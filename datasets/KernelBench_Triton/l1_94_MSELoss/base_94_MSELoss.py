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
        x_ptr,
        y_ptr,
        out_buf_ptr,
        n_elements,
        chunk_size,
        BLOCK_SIZE: tl.constexpr,
        NUM_PIDS: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)

        acc = 0.0
        arange = tl.arange(0, BLOCK_SIZE)
        tl.multiple_of(arange, 128)

        start = pid * chunk_size
        end_val = tl.minimum(start + chunk_size, n_elements.to(tl.int64))

        offset = start
        while offset < end_val:
            offsets = offset + arange
            tl.max_contiguous(offsets, BLOCK_SIZE)
            mask = offsets < end_val
            x_val = tl.load(x_ptr + offsets, mask=mask, other=0.0)
            y_val = tl.load(y_ptr + offsets, mask=mask, other=0.0)
            diff = (x_val - y_val).to(tl.float32)
            acc += tl.sum(diff * diff, axis=0)

            offset += BLOCK_SIZE

        tl.store(out_buf_ptr + pid, acc)
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

        BLOCK_SIZE = 16384
        NUM_PIDS = 512
        num_pids = min(NUM_PIDS, triton.cdiv(n, BLOCK_SIZE))
        chunk_size = triton.cdiv(n, num_pids)

        if (self._acc_buf is None) or (self._acc_buf_device != dev) or (self._acc_buf.shape[0] != num_pids):
            self._acc_buf = torch.zeros(num_pids, device=dev, dtype=torch.float32)
            self._acc_buf_device = dev
        else:
            self._acc_buf.zero_()

        grid = lambda META: (num_pids,)
        _mse_partial_sum_kernel[grid](
            x, y, self._acc_buf, n, chunk_size,
            BLOCK_SIZE=BLOCK_SIZE,
            NUM_PIDS=num_pids,
            num_warps=4,
            num_stages=3,
        )

        total = self._acc_buf.sum()
        mean = total / n
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
