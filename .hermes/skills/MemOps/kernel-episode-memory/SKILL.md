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
>    `~/CompilerClaw/.hermes/plugins/kernel-episodes/__init__.py` to call a
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

   *** Verify each pattern fits your kernel before adopting (see Pitfalls below) ***

2. Apply optimization  →  see triton-ascend-optimization-patterns

3. Run on cannsim      →  see triton-ascend-cannsim

4. After getting results:
   episode_write(kernel_name="<op>", target="ascend910_9589", ...)
```

## Pitfalls

### Episode patterns are case-specific — verify before adopting

Retrieving an episode is step 1, not step done. Episode patterns describe
*one kernel's journey from v1 to v2*. They rarely transfer 1:1 to a different
kernel, even within the same operator class. Example from June 2026:

- Episode 42 documents that l1_2 (Standard MatMul) v2 won big with
  `tl.static_range` + `al.multibuffer`. The **load-bearing** win was
  `tl.static_range` (it removed 6 of 8 `SET_INTRA_BLOCKI` events). The
  `al.multibuffer` was a smaller follow-up.
- Applied the same combo to l1_1 (Square MatMul): sub-kernel data showed
  -0.6% at K=2 (noise) and **+4.6% REGRESSION at K=4** with new
  `WAIT_FLAG_MTE3@MTE2 = 29,319 cyc` stalls. Why? l1_1's v1 *already had*
  `tl.static_range` (from episode 41), so the multibuffer had nothing to
  pipeline and instead doubled MTE2 buffer pressure + injected sync waits.

**Verify-before-adopt checklist:**

1. **Identify the load-bearing insight** in the episode's `action` and `result`.
   What concrete code change produced the speedup?
2. **Check if your current v1 already has it.** If your v1 already uses
   `tl.static_range`, the multibuffer is a smaller follow-up and may be
   redundant or even hurt.
3. **Build a sub-kernel A/B test** (grid=1, M=BLOCK_M, K=2×BLOCK_K) for the
   proposed change. Look for:
   - **Event-count delta** (counter of `event name` totals). If event counts
     are identical, the change is a pure scheduling hint — the cycle effect
     will be small (sub-1% to a few %).
   - **Per-pipeline busy_cyc delta** on the bottleneck lane.
   - **New CRITICAL events** (e.g. `WAIT_FLAG_*` with high total_cyc) mean
     the change introduced sync overhead.
4. **Adopt only on a clear sub-kernel win** on YOUR kernel's bottleneck.
   Otherwise, document it as "tested but not adopted" with the data, and
   record an episode about the negative finding so future sessions don't
   re-propose it.

**Honest negative results are valuable.** Recording a "tested but not
adopted" episode with the data prevents future agents from blindly applying
the same pattern to similar kernels and re-discovering the regression.
