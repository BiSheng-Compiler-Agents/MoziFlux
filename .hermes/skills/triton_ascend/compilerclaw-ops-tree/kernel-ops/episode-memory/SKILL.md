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

### Step 2: Apply Optimization

Apply patterns from episodes to the kernel. See kernel-ops/codegen and kernel-ops/optimization leaves for implementation details.

### Step 3: Run on cannsim

Use `cannsim_remote_run` to validate. See kernel-ops/simulation leaf.

### Step 4: After Getting Results — Write an Episode

```python
episode_write(
    kernel_name="softmax",
    target="ascend910_9589",
    observation="128 rows x 1024 cols float32. Baseline 1D grid, kernel not resolving.",
    thoughts="I noticed the 1D grid creates too many small programs with high FFTS dispatch cost...",
    action="I rewrote with BLOCK_M=2, BLOCK_N=2048, online max/sum, tl.multiple_of + tl.max_contiguous...",
    result="Resolves at 15866 µs. num_stages=1 correct. Next: try BLOCK_M=4.",
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

## Tools Reference

### `episode_write`
Record a new optimization episode after completing an attempt.
Required: `kernel_name`, `target`, `observation`, `thoughts`, `action`, `result`

### `episode_retrieve`
Full-text search across all episode content fields. Use before starting an optimization.
Required: `query` | Optional: `kernel_name`, `target`, `limit` (default: 5)

Returns episodes formatted as:
```
Retrieved episodes:

===== EPISODE 0 =====
{ "id": 1, "kernel_name": "softmax", ... }
```

> **FTS5 query syntax pitfall.** SQLite FTS5 treats `term:term` as a `column:term` filter.
> A natural query like `element-wise softmax` is tokenized into `element`, `-`, `wise`, `softmax`
> — FTS5 then looks for a column named `wise`, raising `sqlite3.OperationalError: no such column: wise`.
> The same trap fires for queries containing `:` `"` `(` `)` `*` `^`.
>
> If `episode_retrieve` raises a `no such column` error:
> 1. Strip hyphens / colons / quotes from the query and retry
>    (`"element wise softmax"` instead of `"element-wise softmax"`).
> 2. Patch the plugin's `_retrieve_episodes` in
>    `~/CompilerClaw/.hermes/plugins/kernel-episodes/__init__.py` to call a
>    `_sanitize_fts_query()` helper that:
>      - splits the query on whitespace,
>      - strips FTS-special chars (`:` `-` `"` `(` `)` `*` `^`) from each token,
>      - wraps each surviving token in double quotes,
>      - joins with ` OR ` for recall-friendly retrieval.
>    Restart Hermes after patching so the plugin reloads.

### `episode_update`
Correct or extend an existing episode by id.
Required: `id` | Optional: any content field

### `episode_delete`
Remove an episode by id.
Required: `id`

### `episode_list`
Browse all episodes newest first. Filter by `kernel_name` and/or `target`.
Optional: `kernel_name`, `target`, `limit` (default: 20), `offset` (default: 0)

Browse the DB directly:
```bash
sqlite3 ~/CompilerClaw/episodes.db \
  "SELECT id, kernel_name, substr(observation,1,80) FROM episodes ORDER BY id DESC"
```

---

## Constraints
- Always write an episode after EVERY successful optimization — never skip this step
- Episodes IDs 12–40 cover 30 patterns from 198 KernelBench pairs, tagged by kernel_name
- `kernel_name` should be operation type: `matmul`, `softmax`, `rmsnorm`, `layernorm`, `gelu`, `relu`, `maxpool`, `elementwise`, `hingeloss`, `cross_entropy`, `embedding`, `general`
- Write in first person (`I ...`) for all content fields
