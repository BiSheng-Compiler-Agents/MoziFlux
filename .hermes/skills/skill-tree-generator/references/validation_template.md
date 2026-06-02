# Validation Template

The following checks must be executed one by one after generation is complete. **No item may be skipped.**

---

## Validation Checklist

### Check 1: Agent memory file existence

Confirm that the project root has a memory file for the corresponding agent, and that it contains the skill-tree routing protocol.

**Target file** (select based on current agent):

| Agent       | Memory File  | Skill Directory                         |
|-------------|-------------|-----------------------------------------|
| Bitfun      | `AGENTS.md` | `.bitfun/skills/`                       |
| Claude Code | `CLAUDE.md` | `.claude/skills/`                       |
| Codex CLI   | `AGENTS.md` | `.agent/skills/` (or user-defined dir)  |
| Hermes      | `AGENTS.md` | `.hermes/skills/`                       |
| OpenClaw    | `AGENTS.md` | `.openclaw/skills/`                     |
| OpenCode    | `AGENTS.md` | `.opencode/skills/`                     |
| Other agents that support AGENTS.md (Cursor / Aider / Jules, etc.) | `AGENTS.md` | same as above |

Action: Check whether the target memory file already contains the `# CRITICAL — DO NOT SKIP` and `## Routing Protocol` routing protocol content.
If it does not exist, append the following content (replace `{SKILLS_DIR}` with the actual skill directory):

```markdown
# CRITICAL — DO NOT SKIP
You MUST follow this routing protocol BEFORE responding to any user request, no exceptions. This includes the very first message in a conversation.
Failure to read ROOT.md first is a critical error. Do not proceed with any task, tool call, or response until routing is complete.

## Routing Protocol
1. Glob `{SKILLS_DIR}/*-tree/ROOT.md` to discover all skill trees. If glob fails, list `{SKILLS_DIR}` and check each `*-tree` subdirectory for `ROOT.md`.
2. Read every ROOT.md file found
3. Follow the routing logic in ROOT.md to select the correct skill
4. Only after routing is complete, proceed with the user's task

This applies to ALL tasks: research, code, editing, questions — everything.
```

**Pass criteria**: The target memory file already exists and contains the `# CRITICAL — DO NOT SKIP` routing protocol, or it has been successfully appended.

**Recommended approach for repos serving two agent types simultaneously**:
- Put skills uniformly in `.agent/skills/`
- Let Claude Code discover them too via symlink: `ln -s .agent/skills .claude/skills`
- Share the same protocol via symlink: `ln -s AGENTS.md CLAUDE.md`

---

### Check 2: Coverage (capability coverage rate)

- Count every capability in the source skill
- Confirm each capability appears in exactly one leaf node
- Confirm the total count in the SKILL-TREE.md mapping table matches the actual count

**Steps**:
1. List all feature points of the source skill
2. Check one by one whether each appears in a leaf node
3. Verify the statistics in SKILL-TREE.md

**Pass criteria**: Each source capability corresponds to exactly one leaf node; mapping table count is accurate.

---

### Check 3: Reachability

- Trace every route from ROOT.md to a leaf node
- Confirm all leaf files are reachable
- Flag any orphan files

**Steps**:
1. List all routing targets in ROOT.md
2. For each ROUTER.md, list all next hops
3. List all .md files that actually exist
4. Confirm every file is covered by some route

**Pass criteria**: All files are reachable from ROOT.md via routing; no orphan files.

---

### Check 4: Disambiguation (disambiguation completeness)

- Identify keywords that appear in multiple sibling nodes
- For Multi-Skill trees: confirm ROOT.md has a disambiguation rule for every shared keyword

**Steps**:
1. Collect trigger keywords from all leaf nodes
2. Find words that appear in 2+ nodes
3. Check whether ROOT.md's disambiguation rules cover every shared word

**Pass criteria**: Every shared keyword has explicit handling in the disambiguation rules.

---

### Check 5: Depth

- The path from ROOT.md to any leaf node must not exceed 4 levels

**Steps**:
1. For each leaf node, count the ROOT → ... → LEAF level count
2. Confirm maximum depth ≤ 4

**Pass criteria**: All leaf nodes have depth ≤ 4.

---

### Check 6: Keyword Quality

- Each leaf node must have clear, mutually exclusive trigger conditions
- Include bilingual keywords where applicable

**Steps**:
1. Check whether each leaf node has Chinese and English keywords
2. Check whether trigger conditions of sibling nodes are mutually exclusive
3. Confirm there is an "other/default" fallback route

**Pass criteria**: Each leaf node has clear trigger conditions, sibling conditions are mutually exclusive, there is a fallback route.

---

### Check 7: Cross-reference Integrity

- Every "Related Skills" path must resolve to an actually existing file
- Sibling references use `../{sibling}/SKILL.md`, not `./{sibling}/SKILL.md`

**Steps**:
1. Collect the `Related Skills` sections from all leaf nodes
2. Verify each path points to a real existing file
3. Check relative path format

**Pass criteria**: All Related Skills references are valid and path format is correct.

---

### Check 8: Functional Routing Tests

Construct ≥ 5 test cases (3 simple + 2 complex) simulating user prompts:

| # | Type   | Requirement |
|---|--------|-------------|
| 1 | Simple | Use a unique keyword, route directly to a single leaf node |
| 2 | Simple | Use a keyword from another domain |
| 3 | Simple | Use a keyword from a third domain |
| 4 | Complex | Use domain-specific terminology to verify routing precision |
| 5 | Complex | Include conflicting signals to verify signal priority |

**Steps**:
For each test case, manually trace the ROOT → ROUTER → LEAF path.

**Pass criteria**: All test cases route correctly to the expected leaf node.

---

### Check 9: Content Preservation

- Confirm all content from the source skill is retained in the leaf nodes
- Instructions in the source skill have not been lost during splitting

**Steps**:
1. Read the complete content of each source skill
2. Confirm every instruction/formula/code block appears in the corresponding leaf node
3. Pay special attention to numerical values, thresholds, and specific implementation details

**Pass criteria**: All execution-level instructions from the source skill are fully migrated to the leaf nodes.

---

### Check 10: Source Skill Existence

- Confirm every source skill referenced when generating the tree exists and has non-empty content
- Prevent generating stubs pointing to non-existent files

**Steps**:
1. List the source skill paths corresponding to all leaf nodes in the tree
2. Confirm each source skill file exists one by one
3. Confirm source skill file content is non-empty (contains at least valid SKILL.md content)

**Pass criteria**: All source skills exist and have non-empty content.

---

### Check 11: No Stub Files

- Check whether each leaf node contains stub patterns pointing to external files
- An agent loading only the tree must be able to execute the task

**Steps**:
1. Read the SKILL.md of each leaf node
2. Check for the following stub patterns — **any one match marks as failed**:
   - `Read and execute the skill at` + file path
   - `**Full Instructions**: Read` + file path
   - `See {file_path}` or `See {url}` as the only content
   - `Refer to {file_path}` or `For details, see {path}`
   - Contains only title + one-line summary + external link, no substantive executable content
3. For leaf nodes matching a stub pattern: the target file content must be inlined
4. If the target file does not exist: **mark as validation failure**, content must be supplemented manually
5. Check for path references pointing outside the tree (e.g. `.claude/skills/<source-skill>/`, absolute paths pointing to source skill directories) → Grep pattern: `\.claude/skills/(?!.*-tree/)` or similar tree-external path patterns
6. Check for uncleaned "参考文档索引" / "Reference" / "References" sections whose file paths point outside the tree or to non-existent locations
7. For large referenced file directories (e.g. `docs/`), confirm they have been copied into the tree and paths have been replaced with tree-internal relative paths (e.g. `../docs/`)

**Pass criteria**: All leaf nodes are self-contained with complete content, no stub patterns of any kind.

---

### Check 12: Reference File Handling (引用文件处理)

Confirm all external references from source skills have been handled correctly.

**CRITICAL — Do NOT use grep to verify content coverage.** Finding a few unique API names with grep does NOT prove a large reference file was fully inlined. A 779-line API reference can fail this check even when 5 grep hits pass. Use the size-based strategy below instead.

**Steps**:
1. List all external files/directories referenced in source skills
2. For each reference file, determine its size:
   - **≤200 lines AND ≤10KB** (short): Verify full content is present — read the source file in full, then confirm every section/table/code block appears in the corresponding leaf node. No grep shortcuts.
   - **>200 lines OR >10KB** (large): Do NOT claim it was inlined. Verify the file was physically COPIED into the tree directory (use `find` or list the tree). Then verify at least one leaf node references the tree-internal path (e.g. `../../shared/references/filename.md`) with a content map (a description of what is in the file so agents know when to read it).
   - Does not exist → reference deleted, replaced with self-contained instructions

3. Confirm no residual paths pointing to source skill directories in any leaf nodes

**Pass criteria**:
- Short files (≤200 lines): every section of source content verifiably present in leaf
- Large files (>200 lines): file physically exists inside the tree AND a leaf node references it with a tree-internal relative path AND a content description
- No source-skill external paths remain in any leaf

---

## Multi-Skill Additional Checks (Multi-Skill trees only)

### Check M1: Shared Keyword Coverage
- List all capabilities shared by 2+ skills
- Confirm ROOT.md disambiguation rules cover every shared keyword

### Check M2: Cross-Skill Path Non-interference
- Trace 3+ routes from different skills
- Confirm no incorrect routing to other skill subtrees

### Check M3: Shared Leaf Accuracy
- For Shared-identical leaf nodes: confirm instructions are completely identical
- If there is any difference in instructions, split into independent leaf nodes

### Check M4: Cross-Cutting Workflow Coverage
- For each skill, confirm cross-cutting/SKILL.md has at least one workflow involving it
- Confirm cross-cutting lists dependency relationships for all skills

---

## Validation Failure Protocol

If any check fails:
1. Fix the problem directly in the generated files
2. Re-run the failed check to confirm the fix
3. Record the problem type for future improvement
