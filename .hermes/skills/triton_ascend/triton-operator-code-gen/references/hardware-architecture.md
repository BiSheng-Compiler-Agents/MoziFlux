# Ascend NPU Hardware Architecture

## Core Architecture

### AI Core Composition

The compute core of the Ascend NPU is the AI Core. A2/A3 chips typically have **24 AI Cores**.

Each AI Core contains:

| Component       | Count | Function                                           | Dedicated Cache |
|-----------------|-------|----------------------------------------------------|-----------------|
| **Cube Core**   | 1     | Matrix multiplication computation                  | L1 Buffer (1MB) |
| **Vector Core** | 2     | Vector computation (element-wise, reduction, etc.) | UB (192KB)      |

### Core Type Selection

| Operator Type | Core to Use | Method to Get Core Count |
|---------------|-------------|--------------------------|
| Pure vector computation (element-wise, reduction) | Vector Core | `get_npu_vectorcore_num()` |
| Matrix multiplication (`tl.dot`) | AI Core | `get_npu_aicore_num()` |
| CV hybrid operators | AI Core + Vector Core | `get_npu_aicore_num()` |

**Cube Core matrix throughput is far higher than Vector Core element-wise computation (tens of times difference).** Matrix multiplication operators (including GEMV, i.e., GEMM with N=1) **must** use `tl.dot` + AI Core. Degrading matrix multiplication to Vector Core element-wise multiply-add is a severe performance anti-pattern. Even when N=1, you should pad to align with BLOCK_N=16 and then use `tl.dot`.

```python
import torch
import triton.runtime.driver as driver


def get_npu_aicore_num():
    device = torch.npu.current_device()
    return driver.active.utils.get_device_properties(device)["num_aicore"]


def get_npu_vectorcore_num():
    device = torch.npu.current_device()
    return driver.active.utils.get_device_properties(device)["num_vectorcore"]
```
---
## Memory Hierarchy
### Three-Level Storage Architecture
```
┌─────────────────────────────────────────────────────────┐
│                    GM (Global Memory)                   |
│                    DDR Large-Capacity Storage           │
│                    Capacity: GB level                   │
│                    Bandwidth: Lower                     │
└───────────────────────────┬─────────────────────────────┘
                            │ MTE2/MTE3 instruction transfer
                            ▼
┌─────────────────────────────────────────────────────────┐
│                    On-Chip Memory                       │
├───────────────────────────┬─────────────────────────────┤
│     L1 Buffer (1MB)       │    UB (192KB)               │
│     Cube Core Dedicated   │    Vector Core Dedicated    │
│     Matrix Block Cache    │    Vector Computation Cache |
└───────────────────────────┴─────────────────────────────┘
```

### GM (Global Memory)

- Location: DDR memory
- Capacity: GB level
- Characteristic: Large capacity but high access latency
- Usage: Stores input/output tensors, model parameters

### UB (Unified Buffer)

- Location: Inside AI Core, dedicated to Vector Core
- Capacity: 192KB (A2/A3)
- Characteristic: High-speed cache, low access latency
- Usage:
  - Stores input/output data for vector computation
  - Intermediate results for reduction operations
  - Temporary data for activation function computation

### L1 Buffer
- Location: Inside AI Core, dedicated to Cube Core
- Capacity: Typically 1MB (A2/A3)
- Characteristic: Dedicated cache for matrix computation
- Usage:
  - Stores tiled data for matrix multiplication
  - Intermediate results for QK^T, SV computation

## Data Paths
### Complete Computation Path
```
┌───────────────────────────────────────────────────────────────────────┐
│                         Computation Flow                              │
├───────────────────────────────────────────────────────────────────────┤
│                                                                       │
│   GM ──────► UB ──────► Vector Core ──────► UB ──────► GM             │
│ (input)   MTE2 move   vector compute    MTE3 move   (output)          │
│                                                                       │
│   Or                                                                  │
│                                                                       │
│   GM ──────► L1 ──────► Cube Core ──────► L1 ──────► GM               │
│ (input)     move     matrix multiply     move     (output)            │
│                                                                       │
└───────────────────────────────────────────────────────────────────────┘
```
### Cube-Vector Collaboration (CV Operators)

For operators that require both matrix multiplication and vector computation (e.g., Attention):
```
GM ──► L1 ──► Cube Core ─────► L1/UB ─────► Vector Core ────► UB ──► GM
        │         │              │                 │           │
        │    QK^T compute  result transfer     Softmax    final output
        │         │              │                 │
        └─────────┴──────────────┴─────────────────┘
                    Data Path
```
**Key Points:**
- Cube computation results need to be transferred from L1 to UB
- Vector Core handles vector operations like Softmax
- Use workspace to cache intermediate results

---

## Memory Alignment Requirements
### Alignment Rules

| Scenario               | Alignment   | Requirement Description               |
|------------------------|-------------|---------------------------------------|
| VV (Vector-Vector)	   | 32 bytes	   | Pure vector computation               |
| CV (Cube-Vector)	     | 512 bytes   | Matrix + vector hybrid computation    |
| UB buffer	             | 32 bytes    | All UB allocations                    |
| Single value buffer	   | 32 bytes	   | Reduction results like mean, variance |
| Matrix operation BLOCK | 512-bytes    | For FP16, BLOCK_M/N/K are multiples of 16 (512B ÷ 2B = 256 elements ÷ 16 granularity = 16)                                                     |

## Grid Distribution Strategy

### Merged Grid Distribution

When the logical core count exceeds the physical core count, use TRITON_ALL_BLOCKS_PARALLEL=1 for automatic optimization:
```bash
export TRITON_ALL_BLOCKS_PARALLEL=1
```

##  Data Type Optimization
### Avoid Using INT64

Some Vector Core operations do not support INT64 and will degrade to scalar operations:
| Operation	 | Unsupported Data Types |
|------------|------------------------|
| Vector ADD | int64                  |
| Vector CMP | int64/int32            |

**Solution:** Use FP32 for comparison operations
```python
# Before: int64 comparison, degrades to scalar
cols = tl.arange(0, BLOCK_N)
xbar = tl.where(cols < N, x - mean, 0.0)

# After: Convert to FP32 for comparison
cols_cmp = cols.to(tl.float32)
xbar = tl.where(cols_cmp < N, x - mean, 0.0)
```

--- 

## Common Issues
### UB Overflow

**Symptom:** ub overflow, requires xxxx bits while 1572684 bits available!

**Causes:**
- Excessive data volume in a single loop
- Unreasonable buffer allocation
- Extra overhead from unaligned memory access

**Solutions:**
- Reduce BLOCK_SIZE
- Use intra-core loop partitioning
- Ensure memory access alignment

### coreDim Exceeds Limit

**Symptom:** coreDim=xxxx can't be greater than UINT16_MAX

**Cause:** grid size exceeds 65535

**Solutions:**
- Increase BLOCK_SIZE
- Set TRITON_ALL_BLOCKS_PARALLEL=1
- Use intra-core loops to reduce grid count

### Precision Loss

**Symptom:** Inaccurate results with FP16 input

**Cause:** Insufficient precision in reduction operations

**Solutions:**
- Upcast to FP32 before reduction
- Complete all computations in FP32
- Downcast to output type at the end

---

## Platform Differences (A2/A3 vs 910_95)
| Feature | A2/A3 | 910_95 |
|---------|-------|--------|
| tl.multibuffer default | ✅ | ❌ | 
| auto_bind_sub_block default | ✅ | ❌ | 
| FP8 support | ❌ | ✅ |
| sync_solver | ✅ | ❌ | 
| inject_block_all | ✅ | ❌ |
| overflow_mode="saturate" | Via FP32 intermediate (slower) | Native support |

**Implications:**
- 910_95 requires manual enablement of tl.multibuffer() and tl.parallel(bind_sub_block=True)
- 910_95 has better saturate cast performance (native instructions)
- FP8 quantization scenarios are only available on 910_95