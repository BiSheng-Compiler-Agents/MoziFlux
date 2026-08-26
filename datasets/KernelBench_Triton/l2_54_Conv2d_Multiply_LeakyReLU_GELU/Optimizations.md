# Optimizations — l2_54 Conv2d_Multiply_LeakyReLU_GELU

## 1. Re-tiled the epilogue by `(N*C)` plane
Baseline decomposed every flat element back into a channel:
```python
plane_idx = offsets // HW
c_idx = plane_idx % C
scale = tl.load(m_ptr + c_idx, mask=mask, other=1.0)
```
Optimized code launches one program per contiguous `(n, c)` plane and loops over HW:
```python
pid_nc = tl.program_id(0)
c_idx = pid_nc % C
scale = tl.load(m_ptr + c_idx).to(tl.float32)
base = pid_nc * HW
for hw0 in tl.range(0, HW, BLOCK_HW, num_stages=2):
    hw = hw0 + offs
    x = tl.load(x_ptr + base + hw, mask=hw < HW, other=0.0, care_padding=False)
```
Rationale: channel is loop-invariant for a plane, so one `% C` and one multiplier load replace per-element `DIV`, `REM`, and gathers.

## 2. Reduced launch count and avoided grid-cap pressure
The default output has `N*C=4096` planes and `HW=254*254`. The optimized direct path uses grid `(N*C,)`, while the baseline grid was `ceil(N*C*HW / BLOCK_SIZE)`.
```python
if NC <= _MAX_GRID:
    _fused_scale_lrelu_gelu_nc_loop[(NC,)](...)
else:
    _fused_scale_lrelu_gelu_nc_persistent[(_MAX_GRID,)](..., n_programs=_MAX_GRID)
```
Rationale: fewer programs reduce FFTS dispatch overhead and the persistent fallback preserves legality when `N*C > 65535`.

## 3. Kept contiguous HW memory access
Each plane is stored contiguously in NCHW after convolution, so the optimized offsets are linear:
```python
tl.load(x_ptr + base + hw, mask=mask, other=0.0, care_padding=False)
tl.store(y_ptr + base + hw, y, mask=mask)
```
Rationale: linear loads/stores allow MTE coalescing and avoid 2-D broadcast offset arrays.

## 4. Preserved exact math and fp32 activation path
The optimized epilogue keeps the same exact GELU formula and fp32 intermediate arithmetic:
```python
v = x.to(tl.float32) * scale
v = tl.where(v >= 0.0, v, v * negative_slope)
y = 0.5 * v * (1.0 + tl.erf(v * 0.7071067811865476))
```
Rationale: correctness stays aligned with `torch.nn.functional.gelu` while eliminating only indexing overhead.

## Measured effect
Cannsim diagnostic epilogue trace: `7006 -> 3034` wall cycles (`2.31x`). Remote hardware benchmark: optimized is `6.91x` faster than Baseline Triton1 on the target shape with max absolute diff `0.000426054`.
