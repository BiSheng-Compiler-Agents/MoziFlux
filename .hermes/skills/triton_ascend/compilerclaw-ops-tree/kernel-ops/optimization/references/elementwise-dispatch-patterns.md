# Elementwise Kernel Optimization Patterns

## Two-Path Dispatch (Direct + Persistent)

Elementwise kernels (Swish, Sigmoid, GELU, ReLU, etc.) require two-path dispatch
to handle both small and large inputs without hitting the Ascend FFTS grid cap.

```python
_MAX_PROGRAMS = 65535  # Ascend FFTS grid cap

class ModelNew(nn.Module):
    def forward(self, x):
        n_elements = x.numel()
        n_tiles = triton.cdiv(n_elements, BLOCK_SIZE)
        n_tiles_min = triton.cdiv(n_elements, MIN_BLOCK)

        if n_tiles_min > _MAX_PROGRAMS:
            # Persistent path: grid capped at MAX_PROGRAMS
            n_programs = min(n_tiles, _MAX_PROGRAMS)
            _kernel_persistent[(n_programs,)](x, y, n, n_programs, BLOCK_SIZE=BLOCK_SIZE)
        else:
            # Direct path: one program per tile
            _kernel_direct[(n_tiles)](x, y, n, BLOCK_SIZE=BLOCK_SIZE)
```

**Routing threshold:** Use `cdiv(n_elements, MIN_BLOCK) > _MAX_PROGRAMS`, NOT a raw
element count. If using `@triton.autotune` with multiple `BLOCK_SIZE` configs,
the threshold MUST use the **smallest** BLOCK in the configs.

## Huge Tensor Persistent Offsets

For tensors large enough to exceed ~2 GiB of byte offset, grid-capping alone is not enough:
late persistent tiles can compute wrong addresses if offsets stay int32. Promote the tile base
and arange to int64 inside the persistent kernel, then validate original-shape correctness.

```python
@triton.jit
def _kernel_persistent(x_ptr, y_ptr, n_elements, n_programs, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    n_tiles = tl.cdiv(n_elements, BLOCK_SIZE)
    for tile_id in range(pid, n_tiles, n_programs):
        base = tile_id.to(tl.int64) * BLOCK_SIZE
        offsets = base + tl.arange(0, BLOCK_SIZE).to(tl.int64)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        tl.store(y_ptr + offsets, f(x), mask=mask)
```

Do not replace this with host-side tensor slicing chunks unless you have measured it: multiple
launches can be slower than a single persistent kernel and may hide the offset bug rather than
fixing the kernel.

## Autotune + Persistent = Separate Kernels

The persistent kernel MUST be a separate `@triton.jit` function, NOT the same one
decorated with `@triton.autotune`. The persistent kernel needs `n_programs` as a
runtime arg (not `tl.constexpr`), and uses a fixed `BLOCK_SIZE`.

## Autotune Key Consistency

When using `@triton.autotune(key=['n_elements_pow2'])`, EVERY kernel decorated by
it (both direct and persistent) MUST declare `n_elements_pow2: tl.constexpr` in
its signature. Missing constexpr param causes:
```
RuntimeError: No valid triton configs. NoneType: None
```

This error appears on large shapes that trigger the dispatch path with the missing
param. Small shapes may use a different dispatch path and work fine, making this
bug size-dependent.

## `get_init_inputs()` Returning `[()]`

Some kernels return `[()]` (a list with one empty tuple) from `get_init_inputs()`,
meaning "no constructor args". The host/profile code must detect this:

```python
init = mod.get_init_inputs() if hasattr(mod, "get_init_inputs") else []
if init == [()]:
    init = []
model = mod.ModelNew(*init)
```

Without this check, `ModelNew(*[()])` passes the empty tuple as a positional arg:
```
TypeError: ModelNew.__init__() takes 1 positional argument but 2 were given
```

## Persistent Kernel Tile Loop

In a persistent kernel, the work loop must iterate over **tiles**, not elements:

```python
# Correct
n_tiles = tl.cdiv(n_elements, BLOCK_SIZE)
for tile_id in range(pid, n_tiles, n_programs):
    offsets = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

# Wrong — skips by n_elements instead of n_tiles
for tile_id in range(pid, n_elements, n_programs):
    ...
```

Using `n_elements` instead of `n_tiles` causes each program to skip by the total
element count instead of advancing to the next tile, producing wrong output.

## Exp-Heavy Activations on Huge Tensors

For SELU/Swish/Sigmoid/GELU-like activations where each element performs `exp` or similarly
expensive vector math, `BLOCK_SIZE=8192` is often a better first candidate than 4096 for
large fp32 tensors: it halves program count and amortizes scalar/FFTS startup while keeping
UB pressure acceptable for a few fp32 intermediates.

```python
_BLOCK_SIZE = 8192
n_tiles = triton.cdiv(n_elements, _BLOCK_SIZE)
if n_tiles > _MAX_PROGRAMS:
    _kernel_persistent[(_MAX_PROGRAMS,)](..., n_programs=_MAX_PROGRAMS,
                                         BLOCK_SIZE=_BLOCK_SIZE)
else:
    _kernel_direct[(n_tiles,)](..., BLOCK_SIZE=_BLOCK_SIZE)
```

When cannsim compares different tile sizes, normalize cycles by elements processed before
claiming a win. Example method: compare `baseline_cycles / baseline_elements` against
`optimized_cycles / optimized_elements`, or report both normalized to 4096 elements. The
persistent sub-kernel may show extra `JUMPC`/`VF` overhead, so judge it by normalized
throughput plus full-shape hardware launchability.

For `profile_kernels.py`, include both a direct-path shape and the original oversized shape.
If the editable baseline's direct grid would exceed `65535`, skip that provider/shape before
launching it (print `INFO`/`SKIP`, return `inf`) rather than poisoning the NPU context; compare
optimized against any safe reference/baseline and PyTorch/ACL.

## Cheap Arithmetic Activations on Huge Tensors

For exact elementwise formulas with only simple arithmetic (e.g. `abs + add + div`, clamp/linear
pieces), the same two-path dispatch rule applies, but the performance balance differs from
exp-heavy activations:

- `BLOCK_SIZE=8192` can be a strong persistent-path default because it halves tile count while UB
  pressure remains low. Do **not** assume it is always best for the direct path: for very cheap
  clamp/min/max-style kernels, 8192 can regress small/direct hardware shapes versus the original
  4096 tile because extra per-tile vector work outweighs launch amortization. Preserve or benchmark
  the original direct block independently, and use a larger block only for persistent oversized
  dispatch if direct shapes regress.
- Judge cannsim by **normalized cycles per element/tile**, not raw wall cycles: larger tiles can
  have higher wall cycles while much better throughput.
- Keep exact math unless the problem explicitly allows approximation; vector divide may become
  the dominant instruction and is intrinsic to exact formulas such as Softsign.
- On the original oversized shape, a legal persistent Triton path may be slightly slower than a
  read-only baseline/provider that uses a different implementation. Do not treat this as a P0;
  report it honestly and only use an ACL fallback if the task prioritizes leaderboard latency over
  maintaining a Triton path.
