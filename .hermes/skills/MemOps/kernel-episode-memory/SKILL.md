---
name: kernel-episode-memory
description: >
  SQLite-backed episode memory for recording and retrieving Triton kernel
  optimization experiences on Ascend NPUs. Use episode_write after every
  optimization attempt and episode_retrieve before starting one to recall
  what worked on similar kernels in the past.
tags: [triton, ascend, npu, optimization, memory, episodes]
metadata:
  hermes:
    requires_tools:
      - episode_write
      - episode_retrieve
      - episode_update
      - episode_delete
      - episode_list
    related_skills:
      - triton-ascend-cannsim
      - triton-ascend-optimization-patterns
---

# Kernel Episode Memory

## Episode schema

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

## When to use

**Before optimizing** — retrieve past episodes to recall what worked:
```
episode_retrieve(
    query="softmax wide rows FFTS dispatch overhead",
    kernel_name="softmax",
    target="ascend910_9589",
    limit=3
)
```

**After optimizing** — record the experience:
```
episode_write(
    kernel_name="softmax",
    target="ascend910_9589",
    observation="128 rows x 1024 cols float32. Baseline 1D grid, kernel not resolving.",
    thoughts="I noticed the 1D grid creates too many small programs with high FFTS dispatch cost...",
    action="I rewrote with BLOCK_M=2, BLOCK_N=2048, online max/sum, tl.multiple_of + tl.max_contiguous...",
    result="Resolves at 15866 µs. num_stages=1 correct. Next: try BLOCK_M=4."
)
```

---

## Tools

### `episode_write`
Record a new optimization episode after completing an attempt.
Required: `kernel_name`, `target`, `observation`, `thoughts`, `action`, `result`

### `episode_retrieve`
Full-text search across all episode content fields. Use before starting an
optimization to recall relevant past experiences.
Required: `query` | Optional: `kernel_name`, `target`, `limit` (default: 5)

Returns episodes formatted as:
```
Retrieved episodes:

===== EPISODE 0 =====
{ "id": 1, "kernel_name": "softmax", ... }
```

> ⚠️ **FTS5 query syntax pitfall.** SQLite FTS5 treats `term:term` as a
> `column:term` filter. A natural query like `element-wise softmax` is
> tokenized into `element`, `-`, `wise`, `softmax` — and FTS5 then looks for
> a column named `wise`, raising `sqlite3.OperationalError: no such column: wise`.
> The same trap fires for queries containing `:` `"` `(` `)` `*` `^`.
>
> If `episode_retrieve` raises a `no such column` error:
> 1. Strip hyphens / colons / quotes from the query and retry
>    (`"element wise softmax"` instead of `"element-wise softmax"`).
> 2. Patch the plugin's `_retrieve_episodes` in
>    `~/CompilerClaw/.hermes/plugins/kernel_episodes/__init__.py` to call a
>    `_sanitize_fts_query()` helper that:
>      - splits the query on whitespace,
>      - strips FTS-special chars (`:` `-` `"` `(` `)` `*` `^`) from each token,
>      - wraps each surviving token in double quotes,
>      - joins with ` OR ` for recall-friendly retrieval.
>    Restart Hermes after patching so the plugin reloads.

### `episode_update`
Correct or extend an existing episode by id.
Required: `id` | Optional: any content or namespace field

### `episode_delete`
Remove an episode by id.
Required: `id`

### `episode_list`
Browse all episodes newest first. Filter by `kernel_name` and/or `target`.
Optional: `kernel_name`, `target`, `limit` (default: 20), `offset` (default: 0)

---

## Workflow

```
1. Before optimizing:
   episode_retrieve(query="<kernel type> <problem keywords>", kernel_name="<op>")

2. Apply optimization  →  see triton-ascend-optimization-patterns

3. Run on cannsim      →  see triton-ascend-cannsim

4. After getting results:
   episode_write(kernel_name="<op>", target="ascend910_9589", ...)
```
