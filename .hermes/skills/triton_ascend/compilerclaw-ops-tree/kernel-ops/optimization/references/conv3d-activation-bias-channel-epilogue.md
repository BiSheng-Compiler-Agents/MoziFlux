# Conv3d activation chain + per-channel bias channel-segment epilogue

Use this reference for kernels shaped like:

```text
Conv3d -> ReLU -> LeakyReLU -> exact GELU -> Sigmoid -> per-channel BiasAdd
```

## Recognition

- The convolution is standard `nn.Conv3d` / ACL-supported; keep it on ACL/PyTorch.
- The editable Triton baseline only handles the post-conv NCDHW epilogue and flattens the conv output.
- Baseline derives channel per element, e.g. `((offs // stride_c) % C)`, then loads bias per vector lane.
- cannsim trace typically shows scalar-index dominance: high `SCALAR`/`SCALARLDST`, with `DIV`, `REM`, `SIGNEXT`, `ST_XD_XN_IMM`.

## Optimization pattern

Tile the contiguous conv output by `(N*C, DHW_tile)` so channel is scalar per tile and HW access remains contiguous:

```python
y = self.conv(x).contiguous()
N, C, D, H, W = y.shape
DHW = D * H * W
tiles_per_segment = triton.cdiv(DHW, BLOCK_SIZE)
total_tiles = (N * C) * tiles_per_segment

if total_tiles <= _MAX_GRID:
    _direct[(total_tiles,)](...)
else:
    _persistent[(min(total_tiles, _MAX_GRID),)](...)
```

Device-side mapping:

```python
tile_id = tl.program_id(0)  # or grid-stride tile loop in persistent path
seg = tile_id // tiles_per_segment
tile_in_seg = tile_id - seg * tiles_per_segment
hw = tile_in_seg * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
mask = hw < DHW
base = seg * DHW + hw
c = (seg - (seg // C) * C).to(tl.int32)

x = tl.load(x_ptr + base, mask=mask, other=0.0, care_padding=False).to(tl.float32)
x = tl.maximum(x, 0.0)
# LeakyReLU immediately after ReLU is exact identity.
x = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
x = 1.0 / (1.0 + tl.exp2(-x * 1.4426950408889634))
b = tl.load(bias_ptr + c * bias_stride_c).to(tl.float32)
tl.store(y_ptr + base, x + b, mask=mask)
```

Persistent path must iterate over **tiles**, not elements:

```python
for tile_id in tl.range(pid, total_tiles, tl.num_programs(0), num_stages=2):
    ...
```

## Why it works

Per-lane channel recovery forces scalar divide/remainder and scalar spill traffic. Segment tiling replaces that with one scalar channel calculation per `DHW` tile and leaves the hot data path as contiguous GM loads/stores.

## Observed result

For a representative fp32 epilogue microprobe (`BLOCK_SIZE=2048`, `C=32`, one spatial tile):

| Metric | Flat baseline | Channel-segment optimized |
|---|---:|---:|
| wall cycles | 28,975 | 3,724 |
| SCALAR busy cycles | 24,843 | 613 |
| SCALARLDST busy cycles | 23,083 | 1,106 |

The bottleneck shifted from scalar indexing to MTE3/store wait, which is the expected healthy endpoint after removing per-element scalar channel math.

## Profiling requirements

- Keep `Baseline Triton2` parser-visible when a `base_*.py` exists, but do not read it when the active sandbox says reference files are forbidden.
- Include a forced-persistent unit test by temporarily lowering the grid cap on a modest tensor.
- Pre-skip comparison providers that would exceed the Ascend 65,535 grid cap; do not poison the NPU context before timing the optimized path.
