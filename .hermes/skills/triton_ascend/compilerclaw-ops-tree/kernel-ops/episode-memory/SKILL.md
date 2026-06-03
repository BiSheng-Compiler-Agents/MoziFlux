---
name: episode-memory
description: Record and retrieve Triton kernel optimization episodes on Ascend NPUs.
---

# Kernel Episode Memory [LEAF NODE]

SQLite-backed episode memory for recording and retrieving Triton kernel optimization
experiences on Ascend NPUs. Use `episode_write` after every optimization attempt and
`episode_retrieve` before starting one to recall what worked on similar kernels in the past.

## Episode Schema

| Field | Description |
|---|---|
| `kernel_name` | Operation being optimized: `softmax`, `matmul`, `layer_norm`, etc. |
| `target` | Hardware target: `ascend910_9589`, `ascend950` |
| `observation` | What the kernel looked like and what the problem was |
| `thoughts` | Reasoning that led to the chosen optimization approach |
| `action` | What was changed, how, and in what form |
| `result` | Speedup achieved, what worked, what to try next time |

Write all content fields in first person (`I ...`).

---

## Workflow

### Step 1: Before Optimizing — Retrieve Past Episodes

Query the episode database for relevant patterns before writing any kernel code:

```python
episode_retrieve(
    query="softmax wide rows FFTS dispatch overhead",
    kernel_name="softmax",
    target="ascend910_9589",
    limit=3
)
```

More query examples:
```python
episode_retrieve(query="matmul low cube utilization dot pad static range", target="ascend950")
episode_retrieve(query="softmax wide rows MTE stall online reduction", target="ascend950")
episode_retrieve(query="norm scalar overhead two-pass single-pass reduction", target="ascend950")
episode_retrieve(query="elementwise large N FFTS dispatch persistent grid", target="ascend950")
episode_retrieve(query="relu fp16 maximum VEC slow fp32 RVECEX upcast", target="ascend950")
```

### Step 2: Apply Optimization and Run on cannsim

Apply patterns from episodes to the kernel. See kernel-ops/codegen and kernel-ops/optimization leaves.
Use `cannsim` to validate. See kernel-ops/simulation leaf.

### Step 3: After Getting Results — Write an Episode

```python
episode_write(
    kernel_name="softmax",
    target="ascend950",
    observation="128 rows x 1024 cols float32. Baseline bottleneck: aiv_scalar=85%, 3605 µs.",
    thoughts="I noticed the scalar overhead comes from 2D broadcast pattern...",
    action="I used host-side expand().contiguous() and uniform row*D+col offset.",
    result="3605 µs -> 752 µs (4.8x). aiv_mte2 now bottleneck. Contiguity > data reuse count.",
)
```

**When to write an episode:**
- Any optimization that changed performance by >10%
- Any compiler crash or silent failure with a discovered workaround
- Any pattern that surprised you (good or bad)
- Any shape-dependent dispatch trick

**Episode quality checklist:**
- `observation`: includes the cannsim metric that was the bottleneck (e.g. `aiv_scalar=85%`)
- `thoughts`: includes what alternatives were considered and why rejected
- `action`: includes the exact code change (one-liner or snippet)
- `result`: includes before/after latency numbers and the key insight

---

## Episode Templates by Kernel Type

### Elementwise / Activation Kernels
```python
episode_write(
    kernel_name="relu",  # or "gelu", "sigmoid", "tanh", etc.
    target="ascend950",
    observation="<kernel_type>: <shape>, <dtype>. Baseline bottleneck: <metric>=<value>%, <latency>us.",
    thoughts="<what was tried>, <why>, <what the trace showed>.",
    action="<key code change>.",
    result="<before> -> <after> (<speedup>x). Key insight: <one-liner>.",
)
```

### Reduction Kernels (Softmax, Norm, Loss)
```python
episode_write(
    kernel_name="softmax",  # or "layernorm", "rmsnorm", "cross_entropy", etc.
    target="ascend950",
    observation="<kernel_type>: <shape>. Baseline: <pass_count>-pass, <metric>=<value>%, <latency>us.",
    thoughts="<e.g. two-pass loads data multiple times, scalar overhead dominates>.",
    action="<e.g. single-pass: load once, tl.sum(x,1) for mean+var, all compute in UB>.",
    result="<before> -> <after> (<speedup>x). <key insight>.",
)
```

### Memory-Bound Kernels (RoPE, Embedding, Gather)
```python
episode_write(
    kernel_name="rope",  # or "embedding", "gather", etc.
    target="ascend950",
    observation="<kernel_type>: <shape>. Baseline: <metric>=<value>%, <latency>us. <N_K> K MTE2 ops.",
    thoughts="<e.g. broadcast stride prevents DMA merging; contiguity > data reuse count>.",
    action="<e.g. host-side expand().contiguous(), uniform row*D+col offset in kernel>.",
    result="<before> -> <after> (<speedup>x). <key insight>.",
)
```

### Matrix Multiplication Kernels
```python
episode_write(
    kernel_name="matmul",
    target="ascend950",
    observation="<kernel_type>: <shape>. Baseline: <metric>=<value>%, <latency>us.",
    thoughts="<what the trace showed, what patterns from episodes were tried>.",
    action="<key code change: e.g. tl.range vs while, tl.dot(a,b,acc), compile_hint>.",
    result="<before> -> <after> (<speedup>x). <key insight>.",
)
```

---

## Recording Remote Verification Results

After `remote_verify` completes, record a comprehensive episode that includes
both the optimization journey and the hardware verification outcome:

```python
episode_write(
    kernel_name="<op_type>",
    target="ascend950",
    observation=(
        "Baseline: <bottleneck>, <cannsim_latency>us. "
        "Remote hardware: <test_passed>, <bench_latency>ms vs torch_npu <ref_latency>ms"
    ),
    thoughts=(
        "<patterns tried>, <what worked>, <what didn't>. "
        "Remote verification: <correctness: PASS/FAIL>, <perf_ratio>"
    ),
    action="<key optimizations>. Remote: uploaded via remote_verify, ran profile_kernels.py --test --bench",
    result=(
        "cannsim: <before> -> <after> (<speedup>x). "
        "Hardware: <latency>ms, <ratio>x vs torch_npu. "
        "Test: <PASS/FAIL>."
    ),
)
```

This ensures future agents can learn from both the optimization patterns AND
the hardware verification results.

---

## Tools Reference

### `episode_write`
Record a new optimization episode after completing an attempt.
Required: `kernel_name`, `target`, `observation`, `thoughts`, `action`, `result`

### `episode_retrieve`
Full-text search across all episode content fields. Use before starting an optimization.
Required: `query` | Optional: `kernel_name`, `target`, `limit` (default: 5)

> **FTS5 query syntax pitfall.** SQLite FTS5 treats `term:term` as a `column:term` filter.
> A natural query like `element-wise softmax` is tokenized into `element`, `-`, `wise`, `softmax`
> — FTS5 then looks for a column named `wise`, raising `sqlite3.OperationalError: no such column: wise`.
> The same trap fires for queries containing `:` `"` `(` `)` `*` `^`.
>
> If `episode_retrieve` raises a `no such column` error:
> 1. Strip hyphens / colons / quotes from the query and retry
>    (`"element wise softmax"` instead of `"element-wise softmax"`).

### `episode_update`
Correct or extend an existing episode by id. Required: `id`

### `episode_delete`
Remove an episode by id. Required: `id`

### `episode_list`
Browse all episodes newest first. Optional: `kernel_name`, `target`, `limit`, `offset`

---

## Pitfalls

### Episode patterns are case-specific — verify before adopting

Episode patterns describe *one kernel's journey from v1 to v2*. They rarely transfer 1:1
to a different kernel, even within the same operator class.

**Verify-before-adopt checklist:**

1. **Identify the load-bearing insight** in the episode's `action` and `result`.
2. **Check if your current v1 already has it.** If so, the follow-up optimization may be redundant or harmful.
3. **Build a sub-kernel A/B test** (grid=1, M=BLOCK_M, K=1×BLOCK_K) for the proposed change.
4. **Adopt only on a clear sub-kernel win** on YOUR kernel's bottleneck. Otherwise, record a "tested but not adopted" episode with the data.

**Honest negative results are valuable.** Recording a "tested but not adopted" episode
with the data prevents future agents from blindly re-discovering the same regression.

---

## Constraints

- Always write an episode after EVERY successful optimization — never skip this step
- `kernel_name` should be operation type: `matmul`, `softmax`, `rmsnorm`, `layernorm`, `gelu`, `relu`, `maxpool`, `elementwise`, `hingeloss`, `cross_entropy`, `embedding`, `general`
- Write in first person (`I ...`) for all content fields
