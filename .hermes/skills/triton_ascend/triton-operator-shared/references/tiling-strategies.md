# Tiling Strategy Detailed Explanation

This document details the design methodology for tiling strategies on Ascend NPU.

## Core Idea of Tiling

### Why Tiling is Needed

1. **Limited UB Capacity**: The AI Core's UB is typically only 192KB, incapable of loading all data at once
2. **Increased Parallelism**: Through inter-core tiling, multiple AI Cores can process in parallel
3. **Optimized Memory Access**: Reduce GM access count and increase data reuse rate

### Two Levels of Tiling
```
┌─────────────────────────────────────────┐
│ Global Memory (GM)                      │
│ ┌──────┬──────┬──────┬──────┬──────┐    │
│ │Core 0│Core 1│Core 2│Core 3│ ...  │    │
│ └──────┴──────┴──────┴──────┴──────┘    │
│ Inter-Core Tiling                       │
└─────────────────────────────────────────┘
↓ Each core processes independently
┌─────────────────────────────────────────┐
│ Unified Buffer (UB)                     │
│ ┌──────┬──────┬──────┬──────┐           │
│ │Tile 1│Tile 2│Tile 3│ ...  │           │
│ └──────┴──────┴──────┴──────┘           │
│ Intra-Core Tiling                       │
└─────────────────────────────────────────┘
```

## Inter-Core Tiling Strategies

### Tiling Principles

1. **Independence**: Data processed by each AI Core is independent, avoiding cross-core communication
2. **Load Balancing**: Each core processes approximately the same amount of data
3. **Data Locality**: Contiguous data assigned to the same core to improve cache hit rate

### Common Tiling Patterns

#### Pattern 1: Tiling by Batch Dimension

**Applicable Scenarios**:
- Input shape is [B, D], B is batch dimension, D is feature dimension
- Operator computes independently along D dimension (e.g., LayerNorm, RMSNorm)

**Tiling Method**:
```python
# Input: x[B, D]
# Number of AI Cores: num_cores

# Step 1: Calculate number of batches per core
batch_per_core = ceil(B / num_cores)

# Step 2: Calculate batch range for current core
core_id = get_core_id()  # Get current Core ID
batch_start = core_id * batch_per_core
batch_end = min((core_id + 1) * batch_per_core, B)

# Step 3: Data processed by current core
x_core = x[batch_start:batch_end, :]  # Shape: [batch_per_core, D]
```

**Example:**
```
Input: x[1024, 768]
Number of AI Cores: 8

Tiling Result:
- Core 0: x[0:128, :]    # 128 batches
- Core 1: x[128:256, :]  # 128 batches
- Core 2: x[256:384, :]  # 128 batches
- ...
- Core 7: x[896:1024, :] # 128 batches
```

#### Pattern 2: Tiling by Feature Dimension

**Applicable Scenarios:**
- Input shape is [B, D], need parallelism along D dimension
- Operator has dependencies along B dimension (e.g., certain reduction operations)

**Tiling Method:**
```python
# Input: x[B, D]
# Number of AI Cores: num_cores

# Step 1: Calculate feature dimension size per core
features_per_core = ceil(D / num_cores)

# Step 2: Calculate feature range for current core
core_id = get_core_id()
feature_start = core_id * features_per_core
feature_end = min((core_id + 1) * features_per_core, D)

# Step 3: Data processed by current core
x_core = x[:, feature_start:feature_end]  # Shape: [B, features_per_core]
```

#### Pattern 3: Row-wise Tiling (Matrix Operations)

**Applicable Scenarios:**
- Matrix multiplication: C = A × B
- Input A shape is [M, K], B shape is [K, N]

**Tiling Method:**
```python
# Matrix multiplication: C[M, N] = A[M, K] × B[K, N]
# Tile along M dimension

# Step 1: Calculate number of rows per core
rows_per_core = ceil(M / num_cores)

# Step 2: Calculate row range for current core
core_id = get_core_id()
row_start = core_id * rows_per_core
row_end = min((core_id + 1) * rows_per_core, M)

# Step 3: Current core computation
A_core = A[row_start:row_end, :]  # Shape: [rows_per_core, K]
C_core = A_core @ B               # Shape: [rows_per_core, N]
```
### Load Balancing Handling

When data size is not divisible by the number of cores:
```python
# Method 1: Round up (last core may process less data)
batch_per_core = ceil(B / num_cores)

# Method 2: Dynamic allocation (more balanced)
base_batch = B // num_cores
remainder = B % num_cores

if core_id < remainder:
    # First 'remainder' cores process one more batch
    batch_start = core_id * (base_batch + 1)
    batch_end = batch_start + base_batch + 1
else:
    # Subsequent cores process base_batch batches
    batch_start = remainder * (base_batch + 1) + (core_id - remainder) * base_batch
    batch_end = batch_start + base_batch
```

## Intra-Core Tiling Strategies
### UB Space Calculation
#### Step 1: Determine Data Type Sizes
```python
# Data type sizes (bytes)
type_sizes = {
    'FP16': 2,
    'BF16': 2,
    'FP32': 4,
    'INT8': 1,
    'INT32': 4,
}
```

#### Step 2: List All Buffer Requirements
```python
# Using RMSNorm as an example, FP16 input upcast to FP32 for computation

# Input buffer (FP16)
input_buffer_size = D * type_sizes['FP16']

# Upcast buffer (FP32)
upcast_buffer_size = D * type_sizes['FP32']

# Square buffer (FP32)
square_buffer_size = D * type_sizes['FP32']

# Mean buffer (FP32, requires 32B alignment)
mean_buffer_size = 32  # Even though logically only 4B needed

# Gamma buffer (FP32)
gamma_buffer_size = D * type_sizes['FP32']

# RMS value buffer (FP32, requires 32B alignment)
rms_buffer_size = 32

# Output buffer (FP16)
output_buffer_size = D * type_sizes['FP16']

# Total space
total_buffer_size = (
    input_buffer_size +
    upcast_buffer_size +
    square_buffer_size +
    mean_buffer_size +
    gamma_buffer_size +
    rms_buffer_size +
    output_buffer_size
)
```

#### Step 3: Calculate Amount Processed per Loop
```python
# UB total size
UB_SIZE = 192 * 1024  # 192KB

# Number of batches that can be processed per loop
batch_per_iteration = UB_SIZE // total_buffer_size

# Actual UB space used
actual_ub_used = batch_per_iteration * total_buffer_size

# Check if UB capacity is exceeded
assert actual_ub_used <= UB_SIZE, f"UB space insufficient: {actual_ub_used} > {UB_SIZE}"
```

### Buffer Allocation Strategies
#### Strategy 1: Fixed Allocation

**Applicable Scenarios:**
- Buffer sizes are fixed
- All buffers used simultaneously

**Example:**
```python
# UB space allocation (192KB)
UB_BASE = 0  # UB start address

# Input buffer (64KB)
input_buffer = UB_BASE
input_buffer_size = 64 * 1024

# Intermediate buffer (96KB)
intermediate_buffer = input_buffer + input_buffer_size
intermediate_buffer_size = 96 * 1024

# Output buffer (32KB)
output_buffer = intermediate_buffer + intermediate_buffer_size
output_buffer_size = 32 * 1024
```

#### Strategy 2: Dynamic Allocation

**Applicable Scenarios:**
- Buffer sizes are variable
- Buffers are time-shared

**Example:**
```python

# Phase 1: Load input data
input_buffer = UB_BASE
load_input(input_buffer, size=D*2)  # FP16

# Phase 2: Upcast computation (reuse input buffer)
upcast_buffer = input_buffer  # Reuse same space
cast_to_fp32(input_buffer, upcast_buffer)

# Phase 3: Compute square (need new buffer)
square_buffer = UB_BASE + D*4  # FP32
compute_square(upcast_buffer, square_buffer)
```

### Alignment Handling
#### 32-byte Alignment Calculation
```python
def align_to_32(size):
    """Align size to 32 bytes"""
    return ((size + 31) // 32) * 32

# Example
actual_size = 4  # Logically need 4 bytes
aligned_size = align_to_32(actual_size)  # Actually allocate 32 bytes
```

#### Single Value Buffer Handling
```python
# Single value results from reduction operations (e.g., mean, variance)
# Logically only need 4 bytes (FP32)
# But hardware requires 32-byte alignment

mean_value_size = 4  # Logical size
mean_buffer_size = 32  # Actual allocated size

# Allocate buffer
mean_buffer = allocate_ub(mean_buffer_size)  # Allocate 32 bytes
```

### 512B Alignment (Recommended for Matrix Operations)

The NPU chip is more affinity to 512-byte aligned scenarios. In matrix multiplication:
| Data Type | Number of Elements for 512B | Cube Granularity Multiple |
|-----------|-----------------------------|---------------------------|
| FP16 (2B) | 256 | 16 |
| FP32 (4B) | 128 | 8 | 
| BF16 (2B) | 256 | 16 |

**Optimal BLOCK Sizes (FP16 matrix multiplication):**
- BLOCK_M=128, BLOCK_N=256, BLOCK_K=256: Fills L0A 64KB
- BLOCK_M=256, BLOCK_N=128, BLOCK_K=256: Fills L0B 64KB

**UB Estimation Formula (matrix multiplication):**
```python
def estimate_ub_usage(BLOCK_M, BLOCK_N, BLOCK_K, dtype_size=2):
    a_tile = BLOCK_M * BLOCK_K * dtype_size
    b_tile = BLOCK_K * BLOCK_N * dtype_size
    accumulator = BLOCK_M * BLOCK_N * 4  # FP32
    mask = BLOCK_M * BLOCK_N
    return a_tile + b_tile + accumulator + mask

# Safety factor: actual requirement × 0.8 (compiler has ~15% extra overhead)
```

## Tiling Strategies for Typical Operators
### Case 1: LayerNorm

**Operator Characteristics:**
- Input shape: [B, D]
- Compute mean and variance along D dimension
- Each batch computed independently

**Tiling Strategy:**
1. **Inter-core tiling:** Tile by batch dimension
```python
batch_per_core = ceil(B / num_cores)
```
2. **Intra-core tiling:**
    - Process 1 batch at a time (need complete D dimension to compute mean and variance)
    - UB requirements:
    ```
    Input buffer (FP16): D × 2B
    Upcast buffer (FP32): D × 4B
    Mean buffer (FP32): 32B
    Variance buffer (FP32): 32B
    Gamma buffer (FP32): D × 4B
    Beta buffer (FP32): D × 4B
    Output buffer (FP16): D × 2B
    ```

### Case 2: Softmax

**Operator Characteristics:**
- Input shape: [B, D]
- Compute exp and reduction along D dimension
- Each batch computed independently

**Tiling Strategy:**
1. **Inter-core tiling:** Tile by batch dimension
2. **Intra-core tiling:**
    - If D is small (< 4096), process multiple batches at once
    - If D is large, need block-wise computation (complex)
    - UB requirements:
    ```
    Input buffer (FP16): D × 2B
    Upcast buffer (FP32): D × 4B
    Max value buffer (FP32): 32B
    Sum value buffer (FP32): 32B
    Exp buffer (FP32): D × 4B
    Output buffer (FP16): D × 2B
    ```

### Case 3: Matrix Multiplication

**Operator Characteristics:**
- Input shapes: A[M, K], B[K, N]
- Output shape: C[M, N]
- Need to access B matrix multiple times

**Tiling Strategy:**
1. **Inter-core tiling:** Tile by M dimension (each core computes several rows of output)
2. **Intra-core tiling:**
    - Tile along K dimension, load portions of A and B each time
    - Accumulate partial results
    - UB requirements:
    ```
    A block buffer: tile_m × tile_k × 2B
    B block buffer: tile_k × tile_n × 2B
    C accumulation buffer: tile_m × tile_n × 4B
    ```
#### Large Matrix Optimization: Diagonal Grid Scheduling

**Problem:** Traditional sequential row scheduling (Core 0 processes M row blocks 0-3, Core 1 processes blocks 4-7) causes the same L2 cache line to be accessed by multiple cores in contention.

**Solution:** Distribute blocks diagonally across the M×N grid to reduce conflicts and improve L2 cache hit rate.

**Applicable Conditions:** NUM_BLOCKS_M >= BLOCK_THRESHOLD and NUM_BLOCKS_N >= BLOCK_THRESHOLD (typically BLOCK_THRESHOLD=4-8)

**Implementation:**
```python
for block_idx in range(pid, NUM_BLOCKS, num_cores):
    if NUM_BLOCKS_M >= BLOCK_THRESHOLD and NUM_BLOCKS_N >= BLOCK_THRESHOLD:
        # Diagonal scheduling
        task_m_idx = block_idx % NUM_BLOCKS_M
        task_n_idx = (block_idx // NUM_BLOCKS_M) % NUM_BLOCKS_N
    else:
        # Sequential scheduling
        task_m_idx = block_idx // NUM_BLOCKS_N
        task_n_idx = block_idx % NUM_BLOCKS_N
```

## Tiling Strategy Design Checklist
### Inter-core Tiling Checks

- [ ] Tiling dimension chosen appropriately (considering data independence)
- [ ] Load balanced (similar amount of data per core)
- [ ] No cross-core communication (each core completes tasks independently)
- [ ] Diagonal scheduling used for large matrices (BLOCK_THRESHOLD check)
- [ ] Boundary handling correct (data range for last core)

### Intra-core Tiling Checks

- [ ] All buffers listed
- [ ] Total buffer size < UB total size
- [ ] Single value buffers allocated 32B space
- [ ] Precision conversion strategy clear (whether up/down casting needed)
- [ ] Amount processed per loop calculated correctly

### Alignment Checks

- [ ] UB buffer addresses 32-byte aligned
- [ ] Single value buffers allocated 32B space
- [ ] All buffer sizes account for alignment

### Performance Optimization Checks

- [ ] Reduce GM access count
- [ ] Increase data reuse rate
- [ ] Fully utilize vector computation
- [ ] Avoid unnecessary precision conversions

## Common Errors and Solutions
### Error 1: Insufficient UB Space

**Symptom:** Compilation error ub overflow, requires X bits while Y bits available

**Causes:**
- Total buffer size exceeds UB capacity
- Forgot to account for alignment overhead
- When using 2D Tiling, forgot offset array and mask array overhead (see supplement below)

**Solutions:**
1. Recalculate all buffer sizes
2. Reduce amount of data processed per loop
3. Consider buffer reuse

### Supplement: UB Budget Calculation for 2D Tiling

When using 2D Tiling, besides data buffers, must account for UB overhead of offset arrays and mask arrays:
```python
# 2D Tiling UB budget formula
# Taking RoPE half mode as example: ROWS_PER_TILE rows × half_D columns

data_buffers = 8 * ROWS * half_D * elem_size  # x1,x2,cos1,cos2,sin1,sin2,out1,out2
offset_arrays = 2 * ROWS * half_D * 4          # off_first, off_second (int32)
mask_arrays = ROWS * half_D                     # half_mask (bool)
index_arrays = 6 * ROWS * 4                     # global_rows, b, n, s, c_base, s_base

total_ub = data_buffers + offset_arrays + mask_arrays + index_arrays
assert total_ub < 192 * 1024, f"UB overflow: {total_ub/1024:.1f} KB"
```

**Key Point:** Offset arrays may be the biggest UB overhead (e.g., when ROWS=128, half_D=64, offsets occupy 64KB). This space must be reserved when determining ROWS_PER_TILE.

### Error 2: Alignment Error

**Symptom:** Hardware error or performance degradation

**Causes:**
- Buffer address not aligned
- Single value buffer allocated insufficient space

**Solutions:**
1. Use alignment function to calculate addresses
2. Allocate 32B uniformly for single value buffers

### Error 3: Load Imbalance

**Symptom:** Some cores finish early, overall performance degrades

**Causes:**
- Data size not divisible by number of cores
- Unreasonable tiling strategy

**Solutions:**
1. Use dynamic allocation strategy
2. Adjust tiling dimension

### Error 4: Precision Loss

**Symptom:** Inaccurate results with FP16 input

**Causes:**
- Reduction operation not upcast
- Too many accumulations

**Solutions:**
1. Upcast to FP32 before reduction
2. Use algorithms like Kahan summation

## Reference Resources

- [Triton-ascend Programming Optimization Guide](https://github.com/triton-lang/triton-ascend/blob/main/docs/en/migration_guide/performance_guidelines.md)