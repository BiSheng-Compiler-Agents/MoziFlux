# Ascend NPU Triton API Constraints (For Static Review)

This document only contains API constraints that can be identified through code reading.

## 1. Masking — Must Follow

Ascend has **zero tolerance** for out-of-bounds access. During static review, check whether all `tl.load`/`tl.store` have the `mask=` parameter:

```python
# ❌ Missing mask
x = tl.load(x_ptr + offsets)

# ✅ With mask
x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

# ✅ make_block_ptr (handles boundaries automatically)
block_ptr = tl.make_block_ptr(
    base=ptr, shape=(M, N), strides=(stride_m, stride_n),
    offsets=(pid_m * BLOCK_M, 0), block_shape=(BLOCK_M, BLOCK_N), order=(1, 0))
data = tl.load(block_ptr)
block_ptr = tl.advance(block_ptr, (0, BLOCK_N))
```

## 2. BLOCK_SIZE Constraints (Statically Checkable)
| Check Item | How to Identify in Code |
|------------|-------------------------|
| BLOCK_SIZE not constexpr | Function parameter missing : tl.constexpr declaration |
| Matrix BLOCK not multiple of 16 | Literals like BLOCK_M=100, BLOCK_N=50 |
| BLOCK_K not aligned | Not calculated according to kalign = 32 // dtype_bytes |

```python
# BLOCK_K alignment (from official test cases)
dtype_bytes = torch.tensor(0, dtype=eval('torch.' + dtype)).element_size()
kalign = 32 // dtype_bytes
BLOCK_K = min(max(K, kalign), 32)
```

## 3. Precision Constraints (Statically Checkable)
| Code Pattern | Issue |
|--------------|-------|
| tl.sum(x_fp16, ...) without preceding .to(tl.float32) | Reduction without upcasting |
| tl.dot(a, b) without explicit out_dtype | fp32 default for floats, only int32 available for int8; explicit specification not required |
| tl.exp(x) instead of tl.exp(x - max_x) | Softmax numerically unstable |

**Matrix Multiplication Precision Pattern (from official test cases):**
```python
if dtype == "int8":
    accumulator_type = tl.int32
else:
    accumulator_type = tl.float32

accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=acc_dtype)
for k in range(0, tl.cdiv(K, BLOCK_K)):
    a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
    b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
    accumulator = tl.dot(a, b, accumulator, out_dtype=acc_dtype)
    a_ptrs += BLOCK_K * stride_ak
    b_ptrs += BLOCK_K * stride_bk
c = accumulator.to(c_ptr.dtype.element_ty)
```

## 4. Grid Configuration Constraints (Statically Checkable)
| Code Pattern | Issue |
|--------------|-------|
| Literals like grid = (20,) | Hardcoded core count |
| Using num_vectorcore for matrix kernel | Kernels with tl.dot must use AI Core |
| Matrix multiplication without tl.dot (implementing matmul/GEMV with element-wise multiply-add) | Cube Core throughput far exceeds Vector Core; even for N=1, must pad and use tl.dot |
| Using num_aicore for element-wise kernel | Kernels without tl.dot must use Vector Core |

```python
import triton.runtime.driver as driver
device = torch.npu.current_device()
# Contains tl.dot
num_aicore = driver.active.utils.get_device_properties(device)["num_aicore"]
# Does not contain tl.dot
num_vectorcore = driver.active.utils.get_device_properties(device)["num_vectorcore"]
```

## 5. Atomic Operation Constraints (Statically Checkable)
| Code Pattern | Issue |
|--------------|-------|
| for ...: tl.atomic_cas/or/xor/and/xchg(...) | Not supported in loops |
| ret = tl.atomic_add(...) and using ret in multi-core kernel | Multi-core add + saving intermediate results not supported |

## 6. Specific Op Constraints (Statically Checkable)

The following constraints are based on actual enablement status from official test cases:

| Op | Constraint | Test Case Status |
|----|------------|------------------|
| tl.dot | Inputs only support int8/fp16/fp32/bf16 | generalization_cases enabled |
| dot_scaled | ⚠ Conditionally supported (lhs/rhs only bf16/fp16, scale only int8 ue8m0, output only fp32) | Conditionally enabled |
| tl.sort | Supports 1D~5D | Both generalization_cases and pytest_ut enabled |
| tl.gather | Supports multiple axes (axis 0~4) | generalization_cases enabled; pytest_ut marked skip |
| permute/trans (2,1,0) | 3D non-adjacent axis transpose | generalization_cases commented out; pytest_ut test_permute_full enabled |
| permute/trans | Does not support int64 | generalization_cases enabled but excludes int64 |
| tensor_descriptor | make/load/store must be used together | generalization_cases enabled |

## 7. Code Pattern Constraints (Statically Checkable)
| Code Pattern | Issue |
|--------------|-------|
| for i in range(N): in kernel | When loop count is small and fixed, consider tl.static_range; for large loops, benefit may be unclear or even degrading, should not blindly replace |
| import numpy/import xxx in kernel | Cannot call third-party libraries inside kernel |
| BLOCK_SIZE parameter missing : tl.constexpr | Must be compile-time constant |
| tensor.item() in host loop | CPU-NPU synchronization bottleneck |

## 8. Control Flow Constraints (Statically Checkable)

Triton kernels compile to MLIR structured control flow (scf.for/scf.while) and do not support early exit.
| Code Pattern | Issue |
|--------------|-------|
| return inside for/while loop | Compilation error: "Cannot have return statements inside while or for" (including returns in child functions, transitively checked) |
| break inside for loop | Compilation error: "unsupported AST node type: Break" |
```python
# ❌ return inside loop
for i in range(N):
    if cond:
        return val  # Compilation error

# ❌ break inside loop
for i in range(N):
    if cond:
        break  # Compilation error

# ✅ Alternative 1: while + boolean flag
i = 0
active = True
while i < N and active:
    result = tl.where(cond, desired, result)
    active = tl.where(cond, False, True)  # Manual termination
    i += 1

# ✅ Alternative 2: tl.where mask (recommended)
for i in range(N):
    result = tl.where(active_mask, compute(i), result)
```

## 9. Tensor Indexing Constraints (Statically Checkable)

Triton tensors do not support Python-style [] subscript operations (both reading and assignment are unsupported).
| Code Pattern | Issue |
|--------------|-------|
|tensor[i] = val | AssertionError (ctx not Load in visit_Subscript) |
| val = tensor[i] | AssertionError |
| tensor[i:j] slicing | Compilation error, Python slicing not supported |
```python
# ❌ Index assignment
local_vector = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
local_vector[i] = 123.0  # AssertionError

# ✅ tl.where as alternative
idx = tl.arange(0, BLOCK_SIZE)
local_vector = tl.where(i == idx, 123.0, local_vector)

# ✅ tl.full (more direct)
local_vector = tl.full((BLOCK_SIZE,), 123.0, dtype=tl.float32)

# ❌ Indexing to get value from tensor
val = tensor[i]  # AssertionError

# ✅ tl.gather as alternative
val = tl.gather(tensor, index, axis=0)

# ❌ Slicing
subx = x[1:3]  # Compilation error

# ✅ tl.extract_slice as alternative
subx = tl.extract_slice(x, offsets=(1,), sizes=(2,), strides=(1,))
```

# Ascend Extension APIs
```python
import triton.language.extra.cann.extension as extension

# extract_slice / insert_slice: process large tensors in slices
acc_i = extension.extract_slice(acc, (offset, 0), (BLOCK_M // 4, HEAD_DIM), (1, 1))
acc = extension.insert_slice(acc, acc_i, (offset, 0), (BLOCK_M // 4, HEAD_DIM), (1, 1))

# extension.sort: 2D/3D multi-dimensional sort
x = extension.sort(x, descending=False, dim=1)
```

## 10. Ascend High-Performance APIs
### tl.parallel(bind_sub_block=True)
Distribute post-dot element-wise operations (activation, cast, store) to 2 Vector Cores for parallel execution.
```python
# ✅ Correct usage: parallelize post-dot operations
for s in tl.parallel(0, 2, bind_sub_block=True):
    sub = tl.extra.ascend.extract_slice(acc, (s * HALF, 0), (HALF, N), (1, 1))
    sub = tl.sigmoid(sub)  # activation
    sub = sub.to(out_dtype)
    tl.store(out_ptr + offsets_for(s), sub, mask=mask)
```

Constraints: Requires `extract_slice`/`insert_slice` for sharding; only applicable to element-wise operations after dot.

### tl.load(care_padding=False)

Masked positions return random values instead of other, reducing extra computation overhead, approximately 5-10% performance improvement.
```python
# ✅ Correct usage: downstream does not depend on masked position values
x = tl.load(ptr + offsets, mask=mask, care_padding=False)
# Values at positions where mask is False are undefined, but are masked out in subsequent computation
result = tl.where(mask, x * scale, 0.0)
```
Constraints: Only safe when masked positions are not used downstream (e.g., result is masked again by `tl.where(mask, ...)`).

### tl.cast(overflow_mode=...)
Controls type conversion overflow behavior:
| Parameter | Behavior | Performance |
|-----------|----------|-------------|
| "trunc" (default) | Truncates high bits | Fast |
| "saturate" | Clamps to target type min/max | On A2/A3, goes through FP32, slower |

```python
# ✅ Correct usage: explicitly specify when saturation is needed
y = tl.cast(x, tl.int8, overflow_mode="saturate")  # clamp to [-128, 127]

# ⚠ Default trunc may silently lose data
y = tl.cast(x, tl.int8)  # Large values will be truncated, not clamped
```
Constraints: On A2/A3, saturate goes through FP32 path; be aware of performance impact.

### sync_block_set / sync_block_wait
Cube-Vector synchronization primitives for synchronizing Vector operations after dot with Cube computation.

```python
# ✅ Correct usage: Cube notifies Vector after computation
tl.dot(a, b, acc)
tl.sync_block_set(sender="cube", receiver="vector", event_id=0)

# Vector side waits for Cube completion
tl.sync_block_wait(sender="cube", receiver="vector", event_id=0)
result = acc.to(tl.float16)
```
Constraints:
- event_id range 0-15
- P0: When multiple parallel blocks use sync, event_id must not conflict

### index_select_simd()
Parallel index selection from triton.language.extra.ascend.libdevice, more efficient than gather.
```python
from triton.language.extra.ascend.libdevice import index_select_simd

# ✅ Correct usage: dim cannot be the last dimension, read_shape[dim] must be -1
# src: (M, K), index: (M,), output: (M, N) where N = len(index)
result = index_select_simd(src, index, dim=0)  # dim=0 < ndim-1=1 ✅
```

Constraints:
- dim < ndim - 1 (cannot be the last dimension)
- read_shape[dim] must be -1
- P1: Check whether gather pattern can be replaced with index_select_simd

## Reference Resources
- [Triton-Ascend Official Repository](https://github.com/triton-lang/triton-ascend)
- [API Data Type Support Matrix](ascend-api-dtype-matrix.md)
