# Persistent Mixed-CV State Ownership

Use this reference when a manually scheduled Cube/Vector kernel is correct for one tile or a shallow reduction, but fails when a physical program processes another output tile or when an additional reduction chunk enters the pipeline.

## Separate the ownership axes

A persistent pipeline can carry several independent kinds of state:

1. **Output-tile state** — running maximum/sum, output accumulator, masks, output coordinates.
2. **Chunk state** — per-chunk rescale factors, ping-pong QK/P/PV slots, reduction-chunk metadata.
3. **Task metadata** — delayed stage coordinates stored in a task ring.
4. **Physical ownership state** — which output tile a persistent program owns on each epoch.

Do not use one modulo counter for all four classes unless the lifetime proof shows they are identical.

## Diagnose the first transition, not only a large failure

Construct boundary probes that change one lifetime dimension at a time:

- `physical_core_count` output tiles: at most one tile per persistent program.
- `physical_core_count + 1` tiles: exactly one program receives a second tile.
- `2 * physical_core_count` tiles: every program receives a second tile.
- two reduction chunks versus three chunks: first reuse of a three-slot delayed-state ring.

Add per-output-tile or per-batch error summaries. If only the first tile owned by program 0 is corrupted while the later tile is correct, a later in-flight epoch overwrote earlier state. If every tile first fails at the third chunk, inspect delayed chunk-scale state instead.

## Physical core count and logical grid are different contracts

Persistent stepping and drain boundaries must use the physical AI-core count:

```python
CORE_NUM = driver.active.utils.get_device_properties(device)["num_aicore"]
grid = (min(CORE_NUM, total_tiles),)
```

Inside the kernel, use the same `CORE_NUM` for:

- persistent loop stride;
- prologue/drain boundary calculations;
- output-epoch calculation;
- task-position reconstruction.

A shape-dependent `grid_dim` is the number of launched programs for that invocation; it is not automatically a safe replacement for the physical ownership stride used by a manually designed schedule.

## Concrete buffers versus loop-carried SSA

Choose representation from lifetime semantics, not convenience.

### Output-state buffers

When runtime branches select among several simultaneously live output-state buffers, do not select a `bl.buffer` in one branch and consume the merged value later if backend lowering cannot prove ownership. Prefer branch-local reads, computation, and direct writes to concrete buffers:

```python
if state_slot == 0:
    m = bl.to_tensor(m_buf0)
    l = bl.to_tensor(l_buf0)
    m_new, l_new, p, scale = vector_stage(...)
    bl.to_buffer(m_new, bind_buffer=m_buf0)
    bl.to_buffer(l_new, bind_buffer=l_buf0)
elif state_slot == 1:
    ...
```

If another independent selector is live (for example a three-slot chunk-scale ring), an explicit product dispatch may be required. Keep it structured and documented; do not duplicate unrelated computation.

### Delayed scale state

Small per-chunk values consumed several stages later are often safer as loop-carried SSA tensors:

```python
scale0 = tl.where(slot == 0, scale_new, scale0)
scale1 = tl.where(slot == 1, scale_new, scale1)
scale2 = tl.where(slot == 2, scale_new, scale2)
```

Return and reassign the full tuple at the caller. A helper that returns updated slots is ineffective if the caller ignores the return value. Do not replace proven SSA state with UB buffers without a depth test that reaches the first slot reuse.

## Accumulator multiplicity

The number of accumulator buffers follows the maximum number of concurrently live output-state slots, not merely ping-pong chunk depth. If three output epochs can coexist, mapping state slots 0 and 2 to one accumulator is unsafe unless a lifetime proof shows they never overlap.

Account for every simultaneously live UB allocation before adding accumulators:

```text
QK ping/pong
PV ping/pong
output accumulators
running max/sum
P or layout-conversion temporaries
compiler-generated outlined-vector temporaries
alignment and safety margin
```

If the correct state multiplicity exceeds UB:

1. shorten transient lifetime or publish P inside the concrete branch;
2. reduce tile size as a correctness-first fallback;
3. consider spilling only after quantifying the bandwidth cost;
4. never alias live state merely to satisfy the allocator.

## Task-relative ping-pong selection

A delayed stage must select buffers from the logical task it consumes, not from the current loop clock. Write the offsets explicitly:

```text
V1 logical task = vtask_id - V1_DELAY
V2 logical task = vtask_id - V2_DELAY
```

Use that logical task for P/PV slot parity and chunk-scale slot selection. Subtracting an odd delay flips parity; using raw `vtask_id` can silently read the opposite ping-pong buffer.

Document each stage in a table:

| Stage | Current clock | Logical task | Buffer selector | State selector |
|---|---:|---:|---:|---:|
| QK producer | `t` | `t` | `t % 2` | output epoch |
| Vector softmax | `t` | `t - d1` | `(t-d1) % 2` | task-ring output slot |
| PV/acc consumer | `t` | `t - d2` | `(t-d2) % 2` | delayed scale slot |

## Verification order

After any ownership change, run these gates in order:

1. one output tile, one/two chunks;
2. one output tile, three or more chunks;
3. one tile per physical program;
4. first second-tile reuse boundary;
5. two tiles per program;
6. causal and noncausal modes;
7. every production shape;
8. benchmark only after all correctness gates pass.

Preserve raw maximum and mean errors for each transition. A passing large case does not replace the boundary probes, because the boundary identifies which lifetime contract was repaired.
