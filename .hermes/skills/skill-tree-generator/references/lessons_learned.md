# Lessons Learned

## L1: Routing table categories are not mutually exclusive
**Problem**: Categories in the L1 routing table overlap, causing the same prompt to match multiple routing targets.
**Prevention**: Routing conditions should be as mutually exclusive as possible; use disambiguation rules to handle edge cases.

## L2: Leaf node content is too broad
**Problem**: Leaf nodes contain too many unrelated workflows, defeating the purpose of modularity.
**Prevention**: Each leaf node should handle only one coherent class of tasks; maintain single responsibility.

## L3: Routing observability is insufficient
**Problem**: The agent either noisily reports every routing intermediate step to the user ("I am determining which module you belong to..."), or outputs no routing information at all, making debugging difficult.
**Prevention**: Execute routing silently by default. Provide an on-demand toggle: when the user's prompt contains "routing debug" / "debug routing" / "routing trace", output the compact routing path `[Route] ROOT → ... → [LEAF]`; otherwise output nothing.

## L4: Context is ignored
**Problem**: Routing decisions only look at the current prompt, ignoring the technical-stack context already established in the conversation.
**Prevention**: Every routing node must explicitly require considering the full conversation history.

## L5: Cross-domain workflows are missing
**Problem**: When a user task spans multiple modules, there is no mechanism to combine and execute multiple leaf nodes.
**Prevention**: ROOT.md must include the instruction "if the task spans multiple categories, read multiple leaf nodes in parallel".

## L6: SKILL-TREE.md mapping table is inaccurate
**Problem**: Leaf-node counts are estimated by eye; the SKILL-TREE.md statistics table does not match reality.
**Prevention**: Count every entry in the mapping table after generating it; never estimate by sight.

## L7: cross-cutting update missed when adding a new skill in Mode 3
**Problem**: When adding a new skill, only the subtree and ROOT.md were updated; cross-cutting/SKILL.md cross-skill workflow definitions were not updated.
**Prevention**: Mode 3 Step 7 now explicitly requires updating cross-cutting/SKILL.md.

## L8: Keyword signals lack layering and depth
**Problem**: The ROOT.md signal table only has high-level descriptor words, missing domain-specific terminology; there is no priority when multiple signals conflict.
**Prevention**: Signals are divided into three tiers — T1 (uniquely determining words), T2 (domain-preference words), T3 (cross-domain general words). Higher tiers override lower tiers during routing. root_template.md disambiguation rules include a signal-priority template.

## L9: Shared leaf node instructions are not fully identical
**Problem**: Similar capabilities from two skills were merged into a Shared-identical leaf node, but the actual instruction details differed.
**Prevention**: Only merge into Shared-identical when instructions are completely identical. Any difference means they should be separate Shared-similar independent leaf nodes.

## L10: Stub problem

**Problem**: When generating a skill-tree, leaf nodes are only generated as stubs (title + category summary + link to an external file) rather than inlining complete data or instructions. A single stub renders that capability unusable; multiple leaf nodes simultaneously stubbed breaks the entire path, making a whole capability domain completely unavailable. This is especially dangerous when the skill itself does not exist in the environment and the tree relies entirely on its own content to guide the agent — in that case, stub = missing functionality.

**Root cause**: (1) The generator did not check whether the source skill exists; (2) the "self-contained" requirement did not cover all leaf nodes; (3) the validation phase did not detect stub patterns; (4) reference data in the original skill (feature dictionaries, conversion tables, API specs, etc.) was summarized rather than fully migrated when split into separate leaf nodes.

> **Institutionalized**: Prevention measures for this problem have been incorporated into `references/error_handling.md`. The error severity table defines Fatal/Degraded/Warning three-tier handling strategies; the self-containment rule prohibits stubs; Step R4 immediate validation intercepts stub patterns before the final validation.

## L11: Leaf node instructions weakened (boundary handling lost)

**Problem**: When splitting the original skill, leaf nodes retained the algorithmic skeleton but weakened critical execution details from the original skill (e.g., the specific implementation of boundary clamping, and the clamp behavior for the 5% tolerance became an "optional suggestion"). The original skill said "acceptable to use a 5% tolerance and clamp values", but after splitting it became "do NOT clamp unless very slightly outside" — a semantic reversal that caused boundary values not to be corrected.
**Prevention**: Leaf nodes must retain all **execution-level instructions** from the original skill (specific code, thresholds, clamp behavior); they cannot only retain "high-level guidance". Numerical processing logic in particular must not be weakened in wording, as that changes behavior. During generation, compare every specific instruction in the original skill to verify it has been fully migrated.

## L12: Mode 3 adding a different skill did not convert to Multi-Skill Tree

**Problem**: Initially a single skill was converted to a skill-tree with Mode 1 (Single-Skill structure, ROOT.md using "Step 1: L1 routing"); when a **different new skill** was added via Mode 3 `--update --add`, Step A detected a Single-Skill Tree and went directly to Step B (add capability to the existing skill). This caused the new skill to be incorrectly treated as a capability increment to the existing skill rather than joining as an independent skill subtree.

**Root cause**: Mode 3 Step A's branching logic was missing the "Single-Skill + different skill" case. Step B was designed with the premise of adding capabilities to the **same skill** and is not applicable for adding a completely new skill.

**Prevention**:
1. **Step A2 explicit judgment**: Under a Single-Skill Tree, must further determine whether `--add` is the same skill or a different skill
2. **Step B2 conversion**: When it is a different skill, the Single-Skill tree must first be converted to a Multi-Skill tree (Mode 2 structure) before adding the new skill
3. **Conversion preserves integrity**: The conversion process keeps all existing leaf node content and routing logic unchanged, only adjusting directory structure and ROOT.md format

## L13: Original Single-Skill top-level module directory not deleted after Step B2 conversion

**Problem**: When converting a Single-Skill tree to a Multi-Skill tree, after moving the original top-level module directories (`charts/`, `interactivity/`, etc.) into the `{existing-skill}/` subdirectory, the original module directories at the tree root were not explicitly deleted. As a result, the tree contained two parallel sets of module files — one under `{existing-skill}/` (Multi-Skill routing targets) and one at the tree root (orphan files, not referenced by any route). This causes confusion and may lead the agent to read the orphan files and bypass the correct Phase 1 routing.

**Root cause**: The "move" step in Step B2 may in practice be executed as "copy" (especially when the target subdirectory already exists), and an explicit delete-original-directory step is missing.

**Prevention**:
1. **Step B2 Step 8 explicit deletion**: After conversion, the original module directories at the tree root (`{module1}/`, `{module2}/`, etc.) must be deleted
2. **Validation detection**: Check 3 (Reachability) will flag all orphan files; after conversion there must be no orphan files
3. **Confirm before deletion**: Before deleting, confirm that corresponding files have been fully migrated into the `{existing-skill}/` subdirectory

## L14: Referenced files not inlined/copied, causing leaf nodes to depend on external source skill

**Problem**: The source skill referenced external files (e.g. `docs/`, `references/`, `scripts/`); the generator kept external paths pointing to the source skill directory (e.g. `.claude/skills/xxx/docs/...`) in leaf nodes rather than inlining the content or copying it into the tree. Result:
- After the source skill is deleted, all retrieval/reference paths in the tree become invalid
- The tree is not truly self-contained, violating the self-containment rule

**Root cause**: The generator treated all references the same way — it did not distinguish between "short content that can be inlined" and "large file sets that should be copied". For large file sets like `docs/`, it simply kept the external path reference.

> **Institutionalized**: Prevention measures for this problem have been incorporated into the Reference File Processing Flow in `references/error_handling.md`. Steps R1-R4 define quantitative standards (≤200 lines / ≤10KB inline, >5 files / >50KB copy) and immediate validation steps.

## L16: `skill_view` fails due to name ambiguity

**Problem**: Calling `skill_view(name="hermes-plugin-development")` returns `"Ambiguous skill name: 2 skills match"` and lists the matching paths, causing the source skill to be unreadable.

**Root cause**: The same skill name exists in multiple directories (e.g. `~/.hermes/skills/autonomous-ai-agents/` and the project-local `DevOps/`). `skill_view` refuses to guess and returns an error directly.

**Prevention**:
1. Prefer using the category path format: `skill_view(name="DevOps/hermes-plugin-development")` (the error message's `hint` field provides the correct format)
2. If the category path is still not unique, use `read_file(path="<absolute_path_from_error_message>/SKILL.md")`
3. When Mode 2 Step A starts collecting source skills, if any one returns an Ambiguous error → immediately retry with this priority order; do not continue using the original name

---

## L17: Check 12 grep-based validation falsely passes for large reference files

**Problem**: Check 12 (Reference File Handling) used grep to confirm a few unique API names from a large reference document appeared in leaf nodes. This passed even though the vast majority of the document's content (entire sections, all enumeration values, all error conditions, all compiler flags) was absent. A 779-line API reference was declared "inlined" based on 5 grep hits.

**Root cause**: Grep-based content verification only proves a few strings are present, not that content is complete. For large reference documents (>200 lines), inline = impossible for full content; the correct strategy per Step R2 is to copy the file into the tree, not claim it was inlined.

**Prevention**:
1. For any reference file >200 lines: **never claim it was fully inlined** — the line threshold in Step R2 exists precisely for this reason. Copy it to the tree.
2. Check 12 must verify the **handling strategy**, not just grep for strings: short files (≤200 lines) → confirm full content is present; long files (>200 lines) → confirm the file was **copied to tree** and leaf nodes reference the tree-internal path.
3. When generating the capability matrix (Step A), note which source skills have large reference files and pre-mark them for copying, not inlining.

---

## L15: "Reference document index" sections in leaf nodes not path-replaced or deleted

**Problem**: After leaf nodes were generated, the "参考文档索引" / "Reference Documents" sections copied from the source skill still retained original paths pointing to the source skill's `references/` directory. These files often do not exist in the copy package, and after the source skill is deleted these paths become completely invalid. More seriously, these sections create the false impression for the agent that "there is more detailed information externally", when in fact the content has already been fully inlined into the leaf node.

**Root cause**: The generator copied the "reference document index" sections verbatim when splitting source skill content into leaf nodes, without making the decision to either replace paths (to point to tree-internal copies) or delete (when referenced files do not exist).

> **Institutionalized**: Prevention measures for this problem have been incorporated into Step R3 (Post-Generation Cleanup) in `references/error_handling.md`. R3.1 defines a complete grep pattern library (Chinese and English reference sections + external pointer phrases + tree-external paths); R3.2 defines a three-step path-replacement / deletion / cleanup logic.
