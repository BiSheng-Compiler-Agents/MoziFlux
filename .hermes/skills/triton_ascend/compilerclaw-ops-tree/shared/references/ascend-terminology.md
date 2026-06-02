# Ascend NPU Key Terminology

Only Ascend-specific concepts that the agent may not know are listed.

## Hardware Architecture

### AI Core
- A2/A3 typically have **24 AI Cores**, each containing **1 Cube core + 2 Vector cores**
- Cube Core: Matrix computation (`tl.dot`), uses L1 Buffer
- Vector Core: Vector/element-wise computation, uses UB
- Different instruction sets within cores, but all are SIMD

### Memory Hierarchy

| Level | Capacity | Ownership | Purpose |
|-------|----------|-----------|---------|
| **GM** (Global Memory) | GB level | Globally shared | Store I/O and parameters, high latency |
| **UB** (Unified Buffer) | **192KB** (A2/A3) | Vector Core exclusive | Current compute data, low latency |
| **L1 Buffer** | ~1MB | Cube Core exclusive | Cube Core execution data |

### UB Constraints
- **32-byte alignment** (aligned access yields better performance)
- Single value buffers (mean, variance, etc.) need 32B allocation (even if logically only 4B needed)
- Safe BLOCK_SIZE = `(196608 - 32) / (number of buffers × dtype size) × 0.8`

## Tiling

- **Inter-core tiling**: Distribute tasks across different AI Cores, `grid = number of physical cores`
- **Intra-core tiling**: Process in batches within a single core to fit UB 192KB

## Memory Alignment

| Level | Alignment Requirement |
|-------|----------------------|
| UB buffer | 32 bytes |
| GM buffer | 16 bytes |

```python
# Alignment calculation
aligned_bytes = ((actual_bytes + 31) // 32) * 32
```

**Key Rules**
- Reduction operations must upcast to FP32 (for FP16/BF16 input)
- Matrix multiplication accumulators use FP32
- Data transfer: GM ↔ UB/L1, should reduce GM access count and increase data reuse
- Block alignment: Matrix operation BLOCK_M/N/K must be multiples of 16 (Cube unit granularity)

## HIVM IR Mapping Reference
### Triton Constructs to HIVM IR Mapping
| Triton Construct | HIVM IR | Description | 
|------------------|---------|-------------|
| tl.load() | hivm.load | PIPE_MTE2 (GM→UB) | 
| tl.store() | hivm.store | PIPE_MTE3 (UB→GM) | 
| tl.dot() | Cube mmad | PIPE_M (matrix computation) | 
| Element operations (add/sub/mul) | hivm.vexp, hivm.vabs, hivm.vadd | PIPE_V (vector computation) |
| al.parallel(bind_sub_block=True) | get_sub_block_idx/get_sub_block_num | Distribute to vector cores |
| sync_block_set/wait | hivm.sync_block [SET/WAIT] | Cube-Vector signal synchronization |
| tl.gather() | hivm.gather | Vector gather |
| tl.index_select() | hivm.index_select | Embedding gather |
| tl.sort() | hivm.sort | Hardware-accelerated sort |

### HIVM Instruction Pipeline Mapping
| HIVM Pipe | Corresponding Triton Operations | Description | 
|-----------|---------------------------------|-------------|
| PIPE_S | Scalar computation | Integer division, loop control | 
| PIPE_V | Vector computation | Element-wise ops, reduction, activation | 
| PIPE_M | Matrix computation | tl.dot, matrix multiplication | 
| PIPE_MTE1 | UB ↔ GM | Vector tl.load/tl.store | 
| PIPE_MTE2 | L1 ↔ GM | Cube input loading | 
| PIPE_MTE3 | L1 ↔ GM | Cube result write-back |
