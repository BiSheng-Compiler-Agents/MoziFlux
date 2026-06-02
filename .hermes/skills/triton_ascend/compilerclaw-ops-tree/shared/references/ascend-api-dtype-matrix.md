# Ascend Triton API Data Type Support Matrix

Source: Official documentation + actual test case enablement status verification

## Legend

- ✓ Supported (test case enabled)
- × Not supported
- ✓* bool internally converted to int8 for computation
- ⚠ Conditionally supported (see notes)

## Key Ops for Static Review

The following are the most frequently encountered and error-prone Op data type constraints during review:

### tl.dot (Matrix Multiplication)

| int8 | int16 | int32 | int64 | fp16 | fp32 | bf16 |
|------|-------|-------|-------|------|------|------|
| ✓ | × | × | × | ✓ | ✓ | ✓ |

- **Accumulator type**: `tl.float32` for floating point, `tl.int32` for int8
- `dot_scaled` completely unsupported

### tl.arange

| int32 |
|-------|
| ✓ |

**Only int32 supported**, return value is int32.

### permute / trans

| int8 | int16 | int32 | int64 | fp16 | fp32 | bf16 | bool |
|------|-------|-------|-------|------|------|------|------|
| ✓ | ✓ | ✓ | × | ✓ | ✓ | ✓ | ⚠ |

- Does not support int64
- 3D (2,1,0) non-adjacent axis transpose: enabled in pytest_ut, commented out in generalization_cases ("not support yet: need bisheng support later"), be aware of compatibility when using
- bool: trans not supported

### gather

| fp16 | fp32 | bf16 |
|------|------|------|
| ✓ | ✓ | ✓ |

- `tl.gather` in generalization_cases supports axis 0~4 (multiple axes)
- `extension.gather` in pytest_ut marked skip ("waiting for the compiler to support")

### sort

| int8 | int16 | fp16 | fp32 | bf16 | bool |
|------|-------|------|------|------|------|
| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

- generalization_cases uses `tl.sort` (1D sorting, supports 1D~5D shapes)
- pytest_ut uses `extension.sort` (supports multi-dimensional dim parameter)

### Atomic Ops

| Op | int8 | int16 | int32 | uint32 | int64 | fp16 | fp32 | bf16 |
|----|------|-------|-------|--------|-------|------|------|------|
| atomic_add | ✓ | ✓ | ✓ | ✓ | × | ✓ | ✓ | ✓ |
| atomic_cas | × | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | × |
| atomic_max/min | ✓ | ✓ | ✓ | × | × | ✓ | ✓ | ✓ |

- `atomic_or/xor/and/xchg/cas` **not supported inside loops**
- `atomic_add` does not support multi-core add + saving intermediate results

## Complete Support Matrix (Reference)

### Creation Ops

| Op | int8 | int16 | int32 | int64 | fp16 | fp32 | bf16 | bool |
|----|------|-------|-------|-------|------|------|------|------|
| arange | × | × | ✓ | × | × | × | × | × |
| cat/full/zeros | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| cast | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

### Memory Ops

| Op | int8 | int16 | int32 | int64 | fp16 | fp32 | bf16 | bool |
|----|------|-------|-------|-------|------|------|------|------|
| load/store | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| make_block_ptr | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | × |

### Math Ops

| Op Category | Integer Series | fp16 | fp32 | bf16 |
|-------------|----------------|------|------|------|
| add/sub/mul/div | ✓ | ✓ | ✓ | ✓ |
| cos/sin/exp/log/sigmoid | × | ✓ | ✓ | ✓ |
| sqrt/rsqrt/fma | × | ✓ | ✓ | ✓ |

### Reduction Ops

| Op | int8 | int16 | int32 | int64 | fp16 | fp32 | bf16 |
|----|------|-------|-------|-------|------|------|------|
| sum/max/min | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| argmax/argmin | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| reduce | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

### Scan Ops

| Op | int8 | int16 | int32 | fp16 | fp32 | bf16 |
|----|------|-------|-------|------|------|------|
| associative_scan | ✓ | ✓ | ✓ | ✓ | ✓ | × |
| cumsum/cumprod | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

## Precision Validation Reference Tolerances

Criteria: **MERE < threshold AND MARE < 10 × threshold**

| Data Type | Threshold | MERE Upper Bound | MARE Upper Bound |
|-----------|-----------|------------------|------------------|
| float16 | 2⁻¹⁰ ≈ 9.77e-4 | 9.77e-4 | 9.77e-3 |
| bfloat16 | 2⁻⁷ ≈ 7.81e-3 | 7.81e-3 | 7.81e-2 (compute after converting to float32) |
| float32 | 2⁻¹³ ≈ 1.22e-4 | 1.22e-4 | 1.22e-3 |
| Integer/bool | exact match | exact match | exact match |