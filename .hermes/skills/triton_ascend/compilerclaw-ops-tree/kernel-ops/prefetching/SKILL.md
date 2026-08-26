---
name: prefetching
description: Implement and optimize mixed Cube/Vector Triton-Ascend kernels with explicit al.scope partitioning, bl.alloc ping-pong buffers, synchronization, and staged preloading/prefetch pipelines.
tags: [triton, ascend, prefetching, preloading, cube-vector, pipeline, al, bl]
---

# Triton-Ascend Prefetching / Preloading

Use this skill for kernels that mix Cube work (`tl.dot`) with Vector work such as reductions,
activations, normalization, masking, layout conversion, or stores. The workflow applies when Cube,
Vector, and data-movement stages have explicit producer/consumer dependencies and can overlap across
logical work items.

## Required references

Read these before implementing or reviewing a mixed Cube/Vector schedule:

1. `../../shared/references/triton-api-reference.md`
   - `al.scope`, `al.sync_block_set/wait`, `al.parallel`, `al.fixpipe`, `al.multibuffer`
   - `bl.alloc`, `bl.to_buffer`, `bl.to_tensor`, `bl.subview`
   - launch options and target-specific extension behavior
2. `../../shared/references/hardware-architecture.md`
   - Cube/Vector execution resources, memory hierarchy, alignment, and platform differences
3. `references/persistent-cv-state-ownership.md`
   - required when delayed task rings, multiple state owners, or persistent accumulators are present
4. `references/compiler-errors-and-fixes.md`
   - required when extension kernels fail frontend lowering, BishengIR compilation, memory planning,
     bufferization, or device execution
5. Load additional references from the index at the end only when their trigger condition applies.

## Goal

Transform a serial dependency chain:

```text
load(i) -> Cube A(i) -> Vector B(i) -> Cube C(i) -> Vector D(i) -> store(i)
```

into an overlapped schedule:

```text
clock t: A(t) + B(t-d1) + C(t-d2) + D(t-d3)
```

Correctness comes from explicit ownership and synchronization. Performance comes from overlapping
independent stages without increasing memory pressure or synchronization cost beyond the benefit.

## Preconditions

Before adding prefetching:

- The unscheduled kernel is correct on every dispatch regime.
- Matrix work uses Cube operations rather than Vector multiply-add emulation.
- Shape, dtype, layout, alignment, and tail behavior are explicit.
- A benchmark and correctness harness exist for every supported regime.
- The target platform and compiler options are known.

Do not pipeline an incorrect or unqualified kernel.

## Core invariants

These invariants apply at every stage:

1. **Separate logical identity from physical storage.**
   - Logical item: the work item being processed.
   - Engine clock: the iteration number of a Cube or Vector traversal.
   - Metadata slot: the task-ring record holding logical coordinates.
   - Buffer slot: the physical ping/pong storage location.
   - State slot: the owner of loop-carried statistics or accumulators.
   These values may have different offsets and must not be collapsed into one counter.

2. **Each physical buffer has one writer and one ownership protocol.**
   A producer cannot overwrite a slot until its consumer releases it.

3. **Each dependency has a balanced event protocol.**
   Every set has a matching wait. Slot-free tokens are initialized and drained explicitly.

4. **Persistent state is keyed by logical output ownership.**
   Do not select accumulators, scales, maxima, sums, or output coordinates solely from the engine
   clock unless the dependency table proves they are equivalent.

5. **Prefer one long-lived scope per engine traversal.**
   Put task loops inside Cube and Vector scopes. Use additional scopes only for dependency phases
   that cannot share a traversal and whose allocation roots remain valid. Repeated scope creation
   inside a hot task loop can prevent memory planning and overlap.

6. **Do not change numerical semantics while changing the schedule.**
   Preserve masks, recurrence order, precision, normalization, layouts, and stores during the first
   scheduling pass.

## Step 1 — Write the dependency and ownership table

Create one row per stage before editing the kernel:

| Field | Meaning |
|---|---|
| `name` | Stage name used in code and diagnostics |
| `engine` | Cube or Vector |
| `depends_on` | Producer stages that must complete first |
| `lag` | Logical delay from engine clock to consumed item |
| `metadata_slot` | Rule for selecting logical coordinates |
| `input_slot` | Rule for selecting each physical input buffer |
| `output_slot` | Rule for selecting each physical output buffer |
| `state_owner` | Rule for selecting persistent state |
| `wait_events` | Events consumed before the stage starts |
| `set_events` | Events produced after data is safe to consume or overwrite |

For a stage at clock `t` with lag `d`:

```text
logical_item = t - d
active iff 0 <= logical_item < item_count
```

Derive ring sizes from live ranges, not from the number of stages:

```text
task metadata ring != persistent state ring != physical buffer slot count
```

If the schedule is represented by a host-side table, project values used inside `@triton.jit` as
actual `tl.constexpr(value)` objects. Computed Python globals without constexpr projection may be
rejected by the frontend.

## Stage 1 — Correct explicit Cube/Vector split

Start with a correct split and no overlap.

### Operation placement

| Operation | Placement |
|---|---|
| Matrix multiply, GEMM/GEMV, matrix accumulation | Cube scope |
| Reductions, activation, normalization, masks, scaling, elementwise work | Vector scope |
| Final layout conversion or output store | Engine that owns the final representation |

Do not alternate tiny scopes for every instruction. Group dependency-ordered work by engine and put
synchronization only at real cross-engine handoffs.

### Vector sub-core ownership

Partition Vector work explicitly when automatic sub-core binding is disabled by the extension path:

- query `al.sub_vec_id()` and `al.sub_vec_num()`;
- split an outer dimension so each sub-core owns contiguous, aligned rows or slabs;
- shape loop-carried Vector state per sub-core from initialization;
- use the sub-core offset for global addressing and handoff subviews;
- use extract/insert only when a full tile is genuinely shared between sub-cores.

Do not let two Vector sub-cores update the same state or output rows.

### Memory placement

Choose storage from the consumer:

| Handoff | Typical storage |
|---|---|
| Cube result consumed by Vector | UB, reached through the platform-supported L0C drain path |
| Vector result consumed by Cube | L1 in the layout required by the Cube operand |
| Vector-only temporary | UB |
| Cube-only operand or retained tile | L1/L0 as required by the matrix operation |

Requirements:

- Respect platform alignment and layout requirements.
- Use the real target's supported L0C drain path; do not assume `al.fixpipe` behavior is identical
  across Ascend families.
- When Cube consumes a Vector-produced matrix, validate the physical fractal/layout order rather than
  relying on logical shape alone.
- Allocate buffers outside runtime branches that select them.

### Synchronization edge

For each handoff:

```python
# Producer
produce(buffer)
al.sync_block_set(producer, consumer, event, producer_pipe, consumer_pipe)

# Consumer
al.sync_block_wait(producer, consumer, event, producer_pipe, consumer_pipe)
consume(buffer)
```

Use event IDs only for concurrently live dependencies and keep them in the supported range. Match
pipe arguments to the actual write/read operation.

Stage 1 is complete only when all regimes pass with explicit scopes and synchronization.

## Stage 2 — Ping/pong handoff buffers

After Stage 1 is correct, duplicate only buffers whose producer and consumer can overlap.

For a two-slot handoff:

```text
producer waits slot_free[s]
producer writes slot[s]
producer sets ready[s]
consumer waits ready[s]
consumer reads slot[s]
consumer sets slot_free[s]
```

At traversal boundaries:

```text
prefree: initialize one free token per physical slot
postwait: consume surplus tokens so the next traversal starts from a known count
```

### Slot-selection rule

Select each handoff independently from the identity that owns it:

```text
input_slot  = f(logical producer item, producer clock)
output_slot = g(logical consumer item, consumer clock)
```

A stage may read one parity and write the opposite parity. Never reuse one `sid` merely because two
buffers are accessed in the same function. Verify each producer/consumer edge algebraically and with
boundary items.

### Explicit runtime selection

When runtime selection between concrete `bl.alloc` buffers is required, use explicit branches:

```python
if slot == 0:
    x = bl.to_tensor(buffer0)
else:
    x = bl.to_tensor(buffer1)
```

Do not create Python lists of local buffers or dynamically index Triton tensors.

### Loop-carried state

For each state value, record:

- logical owner;
- initialization point;
- update point;
- final consumer;
- lifetime in task clocks;
- physical storage slot.

Use the smallest state ring that covers the live range. A single accumulator is valid only when the
schedule proves no second logical output becomes live before the first is finalized.

Stage 2 is complete when ping/pong ownership is balanced, correctness passes, and the same-run
benchmark does not regress the supported matrix.

## Stage 3 — Task-skewed preload pipeline

Stage 3 overlaps different stages from different logical items. Keep the stage bodies unchanged and
move only scheduling, metadata, and ownership mechanics.

### Persistent engine structure

Both engines traverse the same deterministic clock. Each engine owns one long-lived scope:

```python
task_end = item_count + pipeline_depth

with al.scope(core_mode="cube"):
    for t in tl.range(0, task_end):
        item_a = t - stage_a_lag
        if 0 <= item_a < item_count:
            run_cube_stage_a(item_a)

        item_c = t - stage_c_lag
        if 0 <= item_c < item_count:
            wait_for_vector_stage_b()
            run_cube_stage_c(item_c)

with al.scope(core_mode="vector"):
    for t in tl.range(0, task_end):
        item_b = t - stage_b_lag
        if 0 <= item_b < item_count:
            wait_for_cube_stage_a()
            run_vector_stage_b(item_b)

        item_d = t - stage_d_lag
        if 0 <= item_d < item_count:
            wait_for_cube_stage_c()
            run_vector_stage_d(item_d)
```

The pseudocode shows ownership, not exact syntax. Use control flow accepted by the target frontend and
verify the lowered IR.

### Prologue and drain

Derive pipeline depth from the largest active lag. For every stage, define:

```text
first active clock
last active clock
metadata validity guard
event initialization count
event drain count
```

Dummy drain clocks must not create new metadata or perform out-of-bounds pointer arithmetic. A stage
may execute only when its delayed logical item is valid.

### Custom scheduling tables

For a reusable implementation, keep scheduling data separate from stage computation:

```text
name | engine | logical lag | task-ring offset | input phase | output phase | waits | sets
```

Use source generation when stage count changes require a different number of explicit SSA metadata
records. Runtime-generic metadata arrays can add dynamic indexing and loop-carried overhead. The
generated Triton source should contain concrete records, selectors, guards, and stage aliases while
stage bodies remain handwritten.

`templates/fa_stage3.py` is a connected worked example. Use it to understand task skew, event
accounting, and state rotation; derive all offsets and event IDs anew for the target kernel.

## Manual synchronization and launch options

Manual scopes and manual event protocols must have one owner. Do not combine hand-written scheduling
with compiler DAG scheduling unless the generated IR proves they compose correctly.

For repeated manual Vector-to-Cube synchronization inside loops, use the launch contract required to
prevent automatic block-sync injection from adding competing edges. In the supported extension path
this is typically:

```python
disable_auto_inject_block_sync=True
```

Treat launch options as target/compiler inputs:

- confirm each option reaches the intended IR or backend metadata;
- do not assume parsed options affect every architecture;
- compile with the real device target when feature gates are involved;
- retain an option only when correctness and IR inspection show the intended effect.

See `references/extension-compile-path.md` for inspection procedures.

### Host grid

Size the launch grid from physical execution resources, not logical item count. When logical work can
exceed the launch limit, use a persistent one-dimensional grid and let each program traverse items by
the physical program count:

```python
grid = (min(num_aicore, total_items),)

# device
pid = tl.program_id(0)
num_programs = tl.num_programs(0)
for item in tl.range(pid, total_items, num_programs):
    ...
```

Obtain the physical core count from device properties and validate the platform's `coreDim` limit.

## Compiler constraints

Load `references/compiler-errors-and-fixes.md` before modifying numerical code in response to an
extension compiler diagnostic. The reported source line may be a downstream verifier symptom rather
than the structural cause.

### Scope placement

If memory planning fails with repeated Cube/Vector scopes:

1. Allocate shared handoff buffers outside runtime branches.
2. Use one long-lived scope per engine.
3. Put task traversal loops inside those scopes.
4. Keep concrete buffer roots visible at every runtime selection.
5. Recompile and inspect the failing pass before changing numerical code.

### Outlined Vector functions

Outlined Vector helpers can reduce instruction and register pressure, but they impose restrictions on
runtime branches, buffer roots, and temporary capacity. Keep helper inputs/outputs explicit and avoid
nested outlined scopes unless the compiler path supports them.

Load `references/outlined-scope-restrictions.md` before introducing or changing outlined helpers.

### Slice operations

`al.extract_slice` and `al.insert_slice` have frontend constexpr and capacity constraints. Load
`references/extract-slice-frontend.md` when adding or changing slice-based Vector code.

## Platform differences

Do not encode one Ascend family's memory path as a universal rule.

Before implementation, determine:

- available Cube and Vector cores;
- UB, L1, and L0 capacities;
- L0C-to-UB drain support;
- matrix operand layout and alignment;
- event and pipe semantics;
- accepted compiler options and feature gates.

Use `hardware-architecture.md` as the source of platform details. Keep platform-specific branches or
references separate from the universal scheduling procedure.

## Correctness verification

### Required test matrix

Cover every dispatch dimension that changes scheduling or ownership:

- shape and tail behavior;
- dtype and accumulation precision;
- mask or mode variants;
- block configuration;
- short and long traversals;
- first item, slot wraparound, state-ring wraparound, and final drain;
- single-item and minimum-depth cases.

### Boundary probes

For each handoff, verify:

```text
producer logical item
producer clock
written physical slot
consumer clock
read physical slot
release event
```

Probe at least:

```text
first valid item
first ping/pong wrap
first metadata-ring wrap
transition between logical outputs
last valid item
all drain clocks
```

### Event accounting

For each event and traversal:

```text
sets == waits
```

Include prefree and postwait operations in the count. An imbalance may surface only at traversal end.

### Numerical checks

Compare against an independent reference and report at least maximum and mean absolute error. Test all
supported regimes before benchmarking.

## Performance verification

Use same-run comparisons with identical inputs and synchronization. Record:

- per-regime latency;
- geometric mean only across a clearly stated common support matrix;
- compiler options and block configuration;
- whether timing is kernel-level or synchronized end-to-end;
- relevant IR changes for scheduling claims.

Do not keep a scheduling change that regresses a required regime unless the dispatch intentionally
selects a different winner for that regime.

Use hardware counters or simulation to decide whether the limiting stage is Cube, Vector, memory, or
synchronization. More buffering is not automatically beneficial.

## Debugging procedure

Use the failure-signature matrix and structural bisect ladder in
`references/compiler-errors-and-fixes.md`. Preserve the last correct numerical implementation while
isolating compiler and ownership failures.

### Compile failure

1. Identify the last failing pass from compiler logs.
2. Inspect allocation roots, scope nesting, runtime branches, and unsupported operations.
3. Reduce to the smallest structure that retains the failure.
4. Change one structural property at a time.

### Device hang

1. Confirm compilation and launch completed.
2. Run the smallest shape in a fresh process under a timeout.
3. Count every event set/wait, including prefree/postwait.
4. Check producer and consumer pipe arguments.
5. Check that each release corresponds to the slot actually consumed.
6. Re-enable stages one dependency edge at a time.

### Silent corruption

1. Compare logical item, metadata slot, buffer slot, and state slot at boundary clocks.
2. Check that producer and consumer use the same physical handoff identity.
3. Verify initialization occurs before the first update.
4. Verify finalization occurs before state reuse.
5. Compare normalized IR before changing arithmetic.

## Validation checklist

### Dependency and ownership

- [ ] Every stage has an engine, lag, validity guard, and logical item.
- [ ] Metadata, buffer, and state slots are derived independently.
- [ ] Every physical buffer has one writer and a release protocol.
- [ ] State-ring capacity covers the complete live range.
- [ ] Prologue and drain clocks cannot access invalid metadata or pointers.

### Memory and layout

- [ ] Every handoff uses memory visible to its consumer.
- [ ] Cube operands use the required physical layout and alignment.
- [ ] UB/L1/L0 capacity is checked for every block configuration.
- [ ] Partial tiles are masked or rejected by the host.

### Synchronization

- [ ] Event IDs are unique for concurrently live dependencies and in range.
- [ ] Producer and consumer use matching event IDs and pipe pairs.
- [ ] Slot-free tokens are initialized and drained.
- [ ] Set/wait counts balance for short, long, and drain traversals.
- [ ] Manual sync launch options are present when required.

### Numerical state

- [ ] Initialization, update, rescale, normalization, and store order are preserved.
- [ ] Reduction and accumulation precision is unchanged.
- [ ] Every output is finalized before its state is reused.
- [ ] Masks and mode-specific semantics are tested independently.

### Verification

- [ ] Static compilation and diff checks pass.
- [ ] Every supported regime passes correctness.
- [ ] Boundary probes cover slot and state wraparound.
- [ ] A same-run performance comparison against the qualified baseline is complete.
- [ ] Scheduling claims are supported by IR, counters, or simulation.

## Common errors

- Using one counter for logical item, metadata ring, buffer phase, and state owner.
- Selecting both an input slot and output slot from one parity without proving their phases match.
- Reusing a buffer before the consumer releases it.
- Omitting prefree or postwait accounting for counting events.
- Allocating a selected buffer only inside one runtime branch.
- Repeating Cube/Vector scopes inside the task loop instead of loops inside persistent scopes.
- Enabling compiler auto scheduling around a manual dependency protocol without inspecting the IR.
- Assuming a launch option affects every architecture.
- Adding ping/pong buffers without checking UB/L1 capacity.
- Benchmarking before all dispatch regimes pass correctness.
- Reporting a geometric mean across different support matrices.

## Templates and references

### Templates

- `templates/fa_native.py` — unscheduled mixed Cube/Vector example.
- `templates/fa_stage1.py` — explicit Cube/Vector split and handoffs.
- `templates/fa_stage2.py` — ping/pong handoff and state-carrying example.
- `templates/fa_stage3.py` — task-skewed persistent Cube/Vector worked example.

Templates are kernel-specific examples, not universal constants. Replace their stages, lags, layouts,
events, state ownership, and block choices from the target kernel's dependency table.

### References

- `references/compiler-errors-and-fixes.md` — Bisheng/extension diagnostics, structural causes,
  fixes, false leads, IR inspection, and minimal bisect procedure.
- `references/persistent-cv-state-ownership.md` — delayed metadata and persistent state ownership.
- `references/extension-compile-path.md` — extension flags, targets, and IR inspection.
- `references/stage2-performance-model.md` — workload-specific performance evidence and tradeoffs.
- `references/extract-slice-frontend.md` — slice frontend constraints and patterns.
- `references/outlined-scope-restrictions.md` — outlined Vector helper restrictions.
- `../../shared/references/triton-api-reference.md` — Triton-Ascend extension API.
- `../../shared/references/hardware-architecture.md` — architecture and memory-path differences.
