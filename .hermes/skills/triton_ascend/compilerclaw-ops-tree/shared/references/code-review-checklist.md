# Triton Operator Static Code Review Checklist

This checklist only includes items that can be judged directly by reading the code.

## Host Side

### Interface Design
- [ ] Has input shape validation (assert)
- [ ] Has data type validation
- [ ] Has device consistency validation (`x.device == y.device`)
- [ ] Has boundary/empty input handling

### Grid Configuration
- [ ] Core count **not hardcoded** (no literals like `grid = (20,)`)
- [ ] Uses AI Core (`num_aicore`) for kernels containing `tl.dot`, otherwise Vector Core
- [ ] Grid size reasonable (1D recommended)

### Block Size
- [ ] BLOCK_SIZE declared as `tl.constexpr`
- [ ] Matrix operation BLOCK_M/N/K are multiples of 16
- [ ] BLOCK_K satisfies alignment (`kalign = 32 // dtype_bytes`)

## Device Side

### Mask Completeness
- [ ] All `tl.load` have `mask=` + `other=` (or use `make_block_ptr`)
- [ ] All `tl.store` have `mask=` (or use `make_block_ptr`)

### Data Type Compliance
- [ ] `tl.dot` inputs only use int8/fp16/fp32/bf16
- [ ] `dot_scaled` not used
- [ ] `permute`/`trans` not using int64
- [ ] `permute`/`trans` 3D (2,1,0) note compatibility

### Precision Handling
- [ ] Upcast to FP32 before reduction operations (`.to(tl.float32)`)
- [ ] `tl.dot` `out_dtype` (fp32 by default for floats, only int32 available for int8; explicit specification optional)
- [ ] Softmax subtracts max value (`tl.exp(x - max_x)`)
- [ ] Convert back to target precision before output

### Atomic Operations
- [ ] `atomic_or/xor/and/xchg/cas` not inside `for` loop body
- [ ] Return value of `tl.atomic_add` not used in multi-core kernels

### Code Patterns
- [ ] No `return` statements inside `for/while` loops (including returns in child functions)
- [ ] No `break` statements inside `for` loops
- [ ] No use of `tensor[i]` indexing operations (read/assign/slice not supported; use `tl.where`/`tl.gather`/`tl.extract_slice` instead)
- [ ] Small, fixed-iteration loops may consider `tl.static_range` (may degrade performance for large loops, not mandatory)
- [ ] No third-party libraries called inside kernel
- [ ] Vectorized computation (not element-wise)

## Performance Hazards

### Memory
- [ ] No redundant GM access (multiple `tl.load` of same pointer)
- [ ] Contiguous memory access (no `tl.arange * stride` skipping)
- [ ] Sufficient data reuse

### Computation
- [ ] Cube BLOCK are multiples of 16 (can be checked with literals)
- [ ] Small loops may consider `tl.static_range` (use cautiously for large loops)
- [ ] Vectorized computation

### Synchronization
- [ ] No `.item()` in host hot path
- [ ] CPU-NPU synchronization minimized
