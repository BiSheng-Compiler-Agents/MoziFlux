# Optimizations

## 1. Batch multiple GroupNorm groups per Triton program

Baseline launches one program per `(batch, group)` tile:

```python
pid = tl.program_id(0)
n = pid // G
g = pid % G
x = tl.load(x_ptr + n * C + g * Cg + offs, mask=offs < Cg)
mean = tl.sum(x, axis=0) / Cg
```

Optimized path processes up to four groups for the same row in one 2D tile:

```python
gidx = group_tile * GROUP_BLOCK + tl.arange(0, GROUP_BLOCK)
offs_c = gidx[:, None] * Cg + offs[None, :]
x = tl.load(x_ptr + base, mask=mask, other=0.0, care_padding=False).to(tl.float32)
mean = tl.sum(x, axis=1) * inv_cg
```

Rationale: target shape has `G=16, Cg=512`; `GROUP_BLOCK=4` keeps live elements at `4*512=2048`, within UB headroom, while reducing target epilogue program count from `1024*16=16384` to `1024*4=4096`.

## 2. Centered variance in the resident UB tile

```python
x_centered = x - mean[:, None]
var = tl.sum(x_centered * x_centered, axis=1) * inv_cg
y = x_centered * tl.rsqrt(var + eps)[:, None]
```

Rationale: all normalization math is done after a single GM load of the tile; no second pass over `x` is introduced. Computing variance from centered values is numerically safer than `E[x^2] - E[x]^2` while retaining a single-pass tile.

## 3. Grid-capped persistent fallback

```python
if total_tiles > 65535:
    _groupnorm_hardtanh_groupblock_persistent_kernel[(65535,)](...)
else:
    _groupnorm_hardtanh_groupblock_kernel[(total_tiles,)](...)
```

Rationale: Ascend `coreDim` is capped at 65,535. The direct path is preserved for normal/target shapes; the persistent path only handles oversized legal inputs and is covered by `profile_kernels.py` with a correctness-only synthetic dispatch test.
