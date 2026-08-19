# Optimizations Applied

## 1. Removed unused `sum_weight.item()` synchronization

The editable baseline passed `float(sum_weight.item())` into the LayerNorm kernel even though the kernel never used the value because `LayerNorm(x + c) == LayerNorm(x)` for scalar `c`.  The optimized kernel removes the scalar argument from the Triton signature and launch.

```python
# baseline launch
_add_layernorm_lastdim_kernel[grid](..., gamma, beta, float(sum_weight.item()), M, N_ROWS, eps, ...)

# optimized launch
_add_layernorm_lastdim_kernel[grid](..., gamma, beta, M, N_ROWS, eps, ...)
```

Rationale: `.item()` creates a CPU/NPU synchronization in the host path and the device parameter was dead.

## 2. Hybrid post-processing dispatch

Small tensors keep the fused Triton LayerNorm + AvgPool3d/GELU path, while medium/default tensors route the standard post-processing chain to ACL/native PyTorch ops.

```python
if x.numel() > _TRITON_POST_MAX_NUMEL:
    x = self.norm(x)
    x = self.avg_pool(x)
    return self.gelu(x)
```

Rationale: ConvTranspose3d, LayerNorm, AvgPool3d, and GELU are standard ACL-covered operations.  For medium/default 3D volumes, the custom Triton epilogue is scalar-index and launch-heavy; native ACL avoids the grid-cap risk and is faster on hardware.

## 3. Grid-cap avoidance for the required default shape

The editable baseline's default LayerNorm/AvgPool custom kernels launch 65,536 CTAs, which is above the safe Ascend grid cap.  The optimized default path avoids those Triton launches entirely.

```python
# profile guard documents the toxic comparison path
if provider in ("Baseline Triton1", "Baseline Triton2") and label == "default_required":
    return "comparison_provider_preskipped_to_avoid_grid_cap_poisoning"
```

Rationale: correctness and benchmark runs must not poison the NPU context with an invalid `coreDim > 65535` launch.
