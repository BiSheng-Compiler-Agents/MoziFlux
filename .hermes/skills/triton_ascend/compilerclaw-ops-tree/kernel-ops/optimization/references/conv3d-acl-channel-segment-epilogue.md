# Conv3d ACL + channel-segment Triton epilogue

Use this reference for kernels shaped like:

```text
Conv3d -> per-channel scale -> activation/tanh -> per-channel multiply/bias -> sigmoid/activation
```

## Recognition

- The convolution is a standard `nn.Conv3d` / ACL-supported primitive; do **not** replace it with scalar Triton direct convolution.
- The editable baseline already calls `self.conv(x)` and then runs a Triton pointwise NCDHW epilogue.
- The baseline flattens the conv output and computes channel per element, e.g. `(offs // DHW) % C`, then loads per-channel parameters with vector indexes.
- cannsim trace shows SCALAR/SCALARLDST dominance, often including per-element `DIV`, `REM`, `SIGNEXT`, `ST_XD_XN_IMM`, and many scalar load/store events.

## Optimization pattern

Keep Conv3d on ACL/PyTorch and only optimize the epilogue. Tile the contiguous conv output by `(N*C, DHW_tile)` so channel is scalar per program:

```python
x = self.conv(x).contiguous()
_, C, D, H, W = x.shape
DHW = D * H * W
total_segments = x.numel() // DHW  # N*C
tiles_per_segment = triton.cdiv(DHW, BLOCK_SIZE)
total_tiles = total_segments * tiles_per_segment
grid = (min(total_tiles, 65535),)
```

Direct path under the grid cap:

```python
pid = tl.program_id(0)
seg = pid // tiles_per_segment
tile_in_seg = pid - seg * tiles_per_segment
hw = tile_in_seg * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
mask = hw < DHW
c = seg % C
base = seg * DHW + hw

x = tl.load(x_ptr + base, mask=mask, other=0.0, care_padding=False)
sf = tl.load(sf_ptr + c)
b  = tl.load(bias_ptr + c)
# epilogue math...
tl.store(out_ptr + base, y, mask=mask)
```

Persistent path for `total_tiles > 65535` must iterate over **tiles**, not elements:

```python
for tile_id in tl.range(pid, total_tiles, tl.num_programs(0), num_stages=2):
    seg = tile_id // tiles_per_segment
    tile_in_seg = tile_id - seg * tiles_per_segment
    ...
```

## Why it works

Per-element channel computation `(offs // DHW) % C` forces scalar div/rem and per-lane parameter addressing. Segment tiling converts channel to one scalar per program, turning parameter loads into scalar loads and leaving the main data path contiguous over `DHW`.

Observed on a Conv3d scale/tanh/multiply/sigmoid epilogue microprobe:

| Metric | Baseline | Channel-segment epilogue |
|---|---:|---:|
| wall cycles | 16900 | 2171 |
| SCALAR busy cycles | 12126 | 567 |
| SCALARLDST busy cycles | 14902 | 1314 |

The optimized trace bottleneck becomes MTE/load-startup dominated rather than scalar-index dominated.

## Cannsim microprobe guidance

- Trace only the epilogue with `grid=(1,)`; full Conv3d is ACL and not the custom-kernel bottleneck.
- If `BLOCK_SIZE=4096` traces are too slow or hit early-exit instability, shrink the cannsim-only compile constants to `BLOCK_SIZE=256`, `DHW=256`, one tile. Report this honestly as a scale-limited microprobe.
- Keep the same indexing pattern being tested: baseline flat `(offs // DHW) % C`; optimized segment/tile mapping.

## Profiling/unit-test requirements

`profile_kernels.py` should cover:

- Exact/default shape and at least small/medium Conv3d shapes.
- Optimized direct path.
- Forced persistent path by temporarily lowering the module grid cap (e.g. `_MAX_GRID = 1`) on a modest tensor.
- Default `C` fast path and generic `C != default` path if the optimized code specializes the common channel count.

Keep parser-visible columns: `PyTorch / ACL`, `Baseline Triton`, `Optimized Triton` (plus `Baseline Triton2` if a read-only `base_*.py` exists).
