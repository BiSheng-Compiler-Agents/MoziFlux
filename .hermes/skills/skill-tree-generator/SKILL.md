---
name: skill-tree-generator
description: Generates, aggregates, and extends modular skill-trees with hierarchical routing. Supports three modes — (1) convert a monolithic skill into a tree with ROOT.md/ROUTER.md/SKILL.md, (2) aggregate multiple skills into a unified cross-domain tree with shared leaves and disambiguation, (3) incrementally update an existing tree by adding new skills. Use when users need to restructure, merge, or extend skills into context-aware, load-on-demand hierarchies.
---

# Skill Tree Generator

## Usage

```
/skill-tree-generator <skill-name-or-skill-path-or-description>
/skill-tree-generator --aggregate skill1,skill2,... [--domain domain-name]
/skill-tree-generator --update <tree-path> --add <skill>
```

| Input Characteristics | Mode | Description |
|----------------------|------|-------------|
| Single skill path/description, no special flag | **Mode 1** | Convert monolithic skill into routing tree |
| `--aggregate skill1,skill2,...` | **Mode 2** | Aggregate multiple skills into unified cross-domain tree |
| `--update <tree-path> --add <skill>` | **Mode 3** | Incrementally update existing tree |

## Overview

Transform monolithic skills into modular, hierarchical skill-trees (ROOT.md → ROUTER.md → SKILL.md) with dynamic routing. Use when:
- A skill has grown too complex and needs modularization
- Multiple distinct workflows exist within a single skill
- Multiple related skills need to be unified under one routing tree
- Cross-domain workflows span multiple skills
- Overlapping capabilities across skills need deduplication
- An existing skill-tree needs new skills or capabilities added

## Strict Conformance

Before creating or modifying tree output, read and follow `references/strict_conformance.md`. Do not use substitute workflows, fast versions, heuristic-only splitting, or partial validation. If full conformance is impractical, stop and report the blocker before continuing.

---

## Mode 1: Single Skill Tree Generation

Generate a routing tree for a single skill.

**Input**: Skill path or description
**Output**: Complete tree structure in `{skill-name}-tree/`

Example:
```
/skill-tree-generator web-development
```

### Mode 1 Step 1: Analyze Input Skill

First, analyze the input skill to identify:

1. **Core domains** - What major functional areas does the skill cover?
2. **Sub-domains** - Within each core domain, what sub-categories exist?
3. **Leaf capabilities** - What specific tasks/endpoints are at the lowest level?
4. **Routing criteria** - What signals distinguish one path from another?

Read the skill content:
```
$ARGUMENTS
```

If `$ARGUMENTS` is a file path, read that file. If it's a description, use it directly.

### Mode 1 Step 2: Design Tree Structure

Based on analysis, design the hierarchy following these principles:

```
skill-tree/
├── ROOT.md                    # L1 routing protocol
├── {module1}/
│   ├── ROUTER.md              # L2 routing logic
│   ├── {submodule}/
│   │   ├── ROUTER.md          # L3 routing logic
│   │   └── {feature}/SKILL.md # Leaf node
└── {module2}/
    └── ...
```

**Design Guidelines:**
- **L1 modules**: Major functional domains (2-5 modules typical)
- **L2 submodules**: Sub-categories within each domain
- **Leaf nodes**: Specific, atomic tasks/capabilities
- **Depth limit**: 3-4 levels maximum for efficiency

### Mode 1 Step 3: Generate ROOT.md

Read `references/root_template.md` and generate `ROOT.md` following the Single-Skill section template.

### Mode 1 Step 4: Generate ROUTER.md Files

For each non-leaf level, read `references/router_template.md` and generate `ROUTER.md` following that template.

**Routing Table Guidelines (supplement to template):**
- Conditions should be mutually exclusive when possible
- Use keyword matching, domain terminology, task patterns
- Include "Other/Default" row for fallback
- Reference context from conversation history

### Mode 1 Step 5: Generate Leaf SKILL.md Files

**Pre-check**: Execute the **Error Severity & Handling Strategy** in `references/error_handling.md` — if the source skill is unavailable (Fatal), report error and stop; if locally missing (Degraded), generate fallback content.

For each leaf node, read `references/leaf_template.md` for the structure template. Extract the complete content from the original skill and fill it into the template. Do NOT replace any content with a file path reference or external link. Every instruction, code example, API reference, and constraint from the original skill must be inlined directly.

**Reference handling**: Execute **Reference File Processing Flow** Step R1-R4 in `references/error_handling.md` (inventory → decide → cleanup → immediate validation).

**Self-containment**: Follow the **Self-Containment Rule** in `references/error_handling.md` — the generated result must be self-contained as a skill tree; inline short references into leaf nodes, large file sets may be copied into the tree and referenced via tree-internal relative paths.

### Mode 1 Step 6: Create Output Structure

Create all files in the target directory. **Small files (≤10KB) use Write tool; large file sets (>5 files or >50KB) use staging + platform-native copy (`cp -a` on Unix/macOS, `robocopy` on Windows; `xcopy /E /I` only as last fallback) — never copy large directories file-by-file with Write. Treat `robocopy` return codes `< 8` as success.**

```
.claude/skills/{skill-name}-tree/     # Claude Code
# or
.agent/skills/{skill-name}-tree/      # Codex CLI / other AGENTS.md-aware agents
├── ROOT.md
├── SKILL-TREE.md              # Directory structure overview
├── GENERATION-REPORT.md       # Required evidence (see Strict Conformance)
├── {module1}/
│   ├── ROUTER.md
│   └── {submodule}/
│       └── SKILL.md
└── ...
```

### Mode 1 Final Step: Validation + Report

1. **Validate**: Execute every check in `references/validation_template.md`. Read the file and run each check sequentially — this is an executable checklist, not informational. Record pass/fail for each. If any check fails, fix the generated files and re-run that check.
2. **Report**: Once all checks pass, create `GENERATION-REPORT.md` in the tree root directory, following the Required Evidence section in `references/strict_conformance.md`. The validation results recorded in step 1 go into this report.

---

## Mode 2: Multi-Skill Aggregate Tree

Generate a routing tree that covers multiple skills. Skills can be from the same domain or from different domains.

**Input**: `--aggregate skill1,skill2,skill3 [--domain domain-name]`
**Output**: Unified tree with shared ROOT.md, each skill as a sub-tree

`--domain` is optional. When omitted, the tree covers cross-domain skills and ROOT.md routes by domain intent. When provided, the tree covers same-domain skills and ROOT.md routes by capability differences within that domain.

Examples:
```
# Cross-domain: coding + writing + security
/skill-tree-generator --aggregate web-dev,technical-writing,security-review

# Same-domain: frontend frameworks
/skill-tree-generator --aggregate react,vue,svelte --domain frontend
```

Multi-skill trees require a two-phase routing: **Phase 1 selects the skill, Phase 2 selects the capability within that skill.**

### Mode 2 Step A: Cross-Skill Capability Collection

For each skill in the `--aggregate` list, independently analyze and extract capabilities.

Then build a comparison matrix:

```
| Capability Group | Skill_A | Skill_B | Skill_C |
|-----------------|---------|---------|---------|
| project | ✓ create,open,save | ✓ create,save | ✗ |
| editing | ✓ add,remove,set | ✗ | ✓ add,remove |
| export  | ✓ render | ✓ export-pdf | ✓ render |
| session | ✓ undo,redo | ✗ | ✓ undo,redo |
```

### Mode 2 Step B: Shared vs Unique Classification

Classify every capability group:

| Category | Definition | Handling |
|----------|-----------|----------|
| **Unique** | Only one skill has this capability | Place directly in that skill's subtree |
| **Shared-similar** | Multiple skills have similar capabilities | Independent leaf nodes each, disambiguation in ROOT.md |
| **Shared-identical** | Capability/instructions completely identical | Merge into shared leaf node, annotate applicable skills |

### Mode 2 Step C: Two-Phase Hierarchy Design

```
{domain}-tree/
├── ROOT.md                        # Phase 1: select skill
├── SKILL-TREE.md                  # Overview with mapping table
├── GENERATION-REPORT.md           # Required evidence (see Strict Conformance)
├── {skill_a}/                     # Skill A full subtree
│   ├── ROUTER.md                  # Phase 2: select capability
│   └── {capability}/SKILL.md
├── {skill_b}/                     # Skill B full subtree
│   ├── ROUTER.md
│   └── {capability}/SKILL.md
├── shared/                        # Shared-identical capabilities (may be empty — see shared/ note below)
│   └── {capability}/SKILL.md      # annotated: applicable to skill_a, skill_b
└── cross-cutting/
    └── SKILL.md                   # Cross-skill workflows
```

### Mode 2 Step C2: Generate Leaf SKILL.md Files

For each leaf node, generate following the full rules in `references/error_handling.md`:

1. **Pre-check**: Step A already confirmed source skill exists → extract complete content directly. If local content is missing → generate `[AUTO-GENERATED FALLBACK]` fallback per Degraded level
2. **Reference file handling**: Execute Steps R1-R4. In Multi-Skill scenarios, reference files from multiple source skills must be handled uniformly. Large file sets (>5 files / >50KB) use staging + platform-native copy; never use Write to copy file by file
3. **Self-containment**: Follow the Self-Containment Rule — the generated skill tree is completely independent from source skills; inline short references, large file sets may use tree-internal relative paths

### Mode 2 Step D: Multi-Skill ROOT.md Generation

ROOT.md must implement two-phase routing. Read `references/root_template.md` Multi-Skill section and generate ROOT.md following that template.

### Mode 2 Step E: Multi-Skill SKILL-TREE.md Overview

Read `references/overview_template.md` Multi-Skill section and generate `SKILL-TREE.md`. The overview must include a skill dimension with the 能力→叶节点映射表 and Skill 覆盖统计.

### Mode 2 Step F: Cross-Cutting Workflows

Generate `cross-cutting/SKILL.md` following `references/cross_cutting_template.md`. This is a **workflow combiner** that:
1. Provides predefined cross-skill workflow definitions
2. Includes a custom workflow fallback mechanism (mandatory, see L7)

### Mode 2 Final Step: Validation + Report

1. **Validate**: Execute every check in `references/validation_template.md`. Read the file and run each check sequentially — this is an executable checklist, not informational. Record pass/fail for each. If any check fails, fix the generated files and re-run that check.
2. **Report**: Once all checks pass, create `GENERATION-REPORT.md` in the tree root directory, following the Required Evidence section in `references/strict_conformance.md`. The validation results recorded in step 1 go into this report.

---

## Mode 3: Update Existing Tree

Add new skills or capabilities to an existing tree.

**Input**: `--update <tree-path> --add <skill-name>`
**Output**: Updated tree files

Example:
```
/skill-tree-generator --update {skills-dir}/coding-tree --add angular
# e.g. Claude Code:  /skill-tree-generator --update .claude/skills/coding-tree --add angular
# e.g. Codex CLI:    /skill-tree-generator --update .agent/skills/coding-tree --add angular
```

### Mode 3 Step A: Detect Single-Skill or Multi-Skill

Read `<tree-path>/ROOT.md` and check:

| Feature | Single-Skill Tree | Multi-Skill Tree |
|---------|-------------------|------------------|
| Routing table heading | "Step 1: L1 Routing" | "Phase 1: Select Skill" |
| Routing target | `./{module}/ROUTER.md` | `./{skill-name}/ROUTER.md` |
| Disambiguation rules | None | Has disambiguation rules section |

### Mode 3 Step A2: For Single-Skill Tree — Same Skill or New Skill?

**This is the critical branch point.** When the existing tree is Single-Skill, determine whether `--add` is adding capabilities to the **same skill** or adding a **different skill**:

1. Read the tree's `SKILL-TREE.md` to identify the existing skill name(s)
2. Compare with the `--add` skill name/domain:
   - **Same skill** (e.g., tree is `web-dev-tree`, adding more `web-dev` capabilities) → **Step B** (add capability to existing skill)
   - **Different skill** (e.g., tree is `web-dev-tree`, adding `react`) → **Step B2** (transform to Multi-Skill, then add new skill)

**How to judge "same" vs "different":**
- If the `--add` skill's domain overlaps significantly with the existing tree's domain AND the skill name matches the tree's original skill → same skill
- If the `--add` skill is a distinct technology/tool/domain with its own capability set → different skill
- When uncertain, treat as **different skill** (safer to restructure early than to force capabilities into wrong module)

**Branch summary:**
- **Single-Skill Tree + Same skill** → execute **Step B** (add capability to existing skill)
- **Single-Skill Tree + Different skill** → execute **Step B2** (convert to Multi-Skill tree, then add new skill)
- **Multi-Skill Tree** → execute **Step C** (add new skill to Multi-Skill tree)

### Mode 3 Step B2: Transform Single-Skill Tree to Multi-Skill Tree

When adding a different skill to a Single-Skill tree, the tree must be restructured from Mode 1 format to Mode 2 format before adding the new skill.

**Current structure (Single-Skill):**
```
{tree}/
├── ROOT.md              # Step 1: L1 Routing → ./{module}/ROUTER.md
├── SKILL-TREE.md
├── {module1}/
│   ├── ROUTER.md
│   └── {leaf}/SKILL.md
└── {module2}/
    └── ...
```

**Target structure (Multi-Skill):**
```
{tree}/
├── ROOT.md              # Phase 1: Select Skill, Phase 2: Select Capability
├── SKILL-TREE.md
├── {existing-skill}/    # Existing Single-Skill modules moved here
│   ├── ROUTER.md        # New: old L1 routing table moved here
│   ├── {module1}/
│   │   ├── ROUTER.md    # Unchanged
│   │   └── {leaf}/SKILL.md
│   └── {module2}/...
├── {new-skill}/         # New skill subtree (generated per Step C)
│   ├── ROUTER.md
│   └── ...
├── shared/              # Shared capabilities directory
├── cross-cutting/       # Cross-skill workflows
│   └── SKILL.md
```

**Transformation steps:**

1. **Identify existing skill name** from SKILL-TREE.md and the tree directory name
2. **Create `{existing-skill}/` subdirectory** and move all existing module directories into it
3. **Create `{existing-skill}/ROUTER.md`** — convert the old ROOT.md's L1 routing table into a skill-level ROUTER.md:
   - The old ROOT.md's `| {category} | Read ./{module}/ROUTER.md |` table becomes the new ROUTER.md's routing table
   - Add `[L2]` level marker
   - Preserve all routing conditions and context hints
4. **Rewrite ROOT.md** to Multi-Skill format (follow Mode 2 Step D template):
   - Phase 1: Select Skill — includes both existing skill and new skill
   - Phase 2: Select Capability — delegates to skill sub-tree ROUTER.md
   - Add disambiguation rules section
   - Add signal priority table
5. **Create `shared/` directory** — initially empty, populated if Step C finds shared capabilities
6. **Create `cross-cutting/SKILL.md`** — follow Mode 2 Step F template, include workflows combining existing + new skill
7. **Update `SKILL-TREE.md`** — rewrite to Multi-Skill format with skill→capability mapping table and coverage stats
8. **Delete original top-level module directories** — after moving modules into `{existing-skill}/`, the original `{module1}/`, `{module2}/`, etc. directories at the tree root must be physically deleted. These are now orphan duplicates that must not remain alongside the new Multi-Skill structure.
9. **Then proceed to add the new skill** — execute Step C (add new skill to Multi-Skill tree) for the `--add` skill

**Important constraints during transformation:**
- **Preserve all existing leaf content** — no SKILL.md files are modified, only moved
- **Preserve all routing logic** — the old ROOT.md's routing conditions must be accurately transferred to `{existing-skill}/ROUTER.md`
- **Do NOT create stubs** — shared/ may be empty initially; only populate if shared capabilities are detected
- After transformation + adding new skill, run full Mode 2 validation

### Mode 3 Step B: Adding a new capability to the SAME skill (Single-Skill tree)

**Prerequisite**: Step A2 has confirmed that the `--add` skill is the same skill as the existing Single-Skill tree.

1. Identify which module/leaf the new capability belongs to
2. Add to the leaf's workflow and examples
3. Update SKILL-TREE.md mapping table
4. Enrich keyword signals if the capability introduces new intent patterns

### Mode 3 Step C: Adding a new skill to existing Multi-Skill tree

1. **Analyze new skill** — extract full capability set. Execute **Error Severity & Handling Strategy** from `references/error_handling.md`: source skill does not exist → Fatal error, stop
2. **Build comparison matrix** against existing skills (same as Mode 2 Step A)
3. **Identify new shared keywords** — any capability overlapping with existing skills
4. **Update ROOT.md** — add new skill route + update 消歧规则 for ALL new shared keywords
5. **Create new skill sub-tree** — `{new-skill}/ROUTER.md` + leaf SKILL.md files. Execute **Reference File Processing Flow** Step R1-R4 from `references/error_handling.md`, following the **Self-Containment Rule**. Large file sets use staging + platform-native copy; never copy large directories file-by-file with Write
6. **Re-check shared leaves** — if new skill has identical capabilities, update shared leaf
7. **Update cross-cutting/SKILL.md** — add cross-skill workflow definitions + update dependencies (L7: most commonly missed step)
8. **Update SKILL-TREE.md** — add rows to mapping table + update coverage stats

### Mode 3 Final Step: Validation + Report

1. **Validate**: Execute every check in `references/validation_template.md`. Read the file and run each check sequentially — this is an executable checklist, not informational. Record pass/fail for each. If any check fails, fix the generated files and re-run that check.
2. **Report**: Once all checks pass, create `GENERATION-REPORT.md` in the tree root directory, following the Required Evidence section in `references/strict_conformance.md`. The validation results recorded in step 1 go into this report.

---

## Templates

Reference templates are available in `references/`:
- `root_template.md` - ROOT.md generation template (Single-Skill + Multi-Skill)
- `router_template.md` - ROUTER.md generation template
- `leaf_template.md` - Leaf SKILL.md generation template
- `cross_cutting_template.md` - Cross-cutting workflows template (Multi-Skill)
- `overview_template.md` - SKILL-TREE.md overview template
- `validation_template.md` - Executable validation checklist (MUST run after generation)
- `error_handling.md` - Error Handling, Reference File Processing, Self-Containment Rule (Unified Specification)
- `lessons_learned.md` - Lessons Learned (L1-L16)

## Example

**Input**: A monolithic "web-development" skill covering frontend, backend, and DevOps.

**Output Structure**:
```
web-development-tree/
├── ROOT.md
├── SKILL-TREE.md
├── GENERATION-REPORT.md
├── frontend/
│   ├── ROUTER.md
│   ├── react/SKILL.md
│   ├── vue/SKILL.md
│   └── css/SKILL.md
├── backend/
│   ├── ROUTER.md
│   ├── api/SKILL.md
│   └── database/SKILL.md
└── devops/
    ├── ROUTER.md
    ├── ci-cd/SKILL.md
    └── docker/SKILL.md
```

---

## Lessons Learned

> **Full content in** `references/lessons_learned.md`. The table below is an index — see that file for problem descriptions, root causes, and prevention steps.

| ID | Topic | One-liner |
|----|-------|-----------|
| L1 | Routing table categories not mutually exclusive | Use disambiguation rules to handle boundary overlap |
| L2 | Leaf node content too broad | Each leaf node maintains single responsibility |
| L3 | Routing observability insufficient | Silent by default; output `[Route]` path on demand |
| L4 | Context ignored | Routing decisions must consider full conversation history |
| L5 | Cross-domain workflows missing | ROOT.md must include instruction to read multiple leaf nodes in parallel |
| L6 | Mapping table inaccurate | Count every entry; never estimate by sight |
| L7 | Mode 3 missed cross-cutting update | Explicitly update cross-cutting/SKILL.md when adding new skill |
| L8 | Keyword signals lack layering | Signals split into T1/T2/T3 tiers; higher tier overrides lower |
| L9 | Shared leaf node instructions differ | Only merge into Shared-identical when instructions are completely identical |
| L10 | Stub problem | Institutionalized → `references/error_handling.md` |
| L11 | Leaf node instructions weakened | Preserve all execution-level instructions (code, thresholds, clamp behavior) |
| L12 | Mode 3 did not convert Single-Skill | Step A2 explicitly judges same vs different skill |
| L13 | Top-level modules not deleted after conversion | Step B2 step 8 explicitly deletes original module directories |
| L14 | Referenced files not inlined/copied | Institutionalized → `references/error_handling.md` |
| L15 | Reference document index sections not cleaned up | Institutionalized → `references/error_handling.md` |
| L17 | Check 12 grep-based validation false positive | Grep proves presence of a few strings, NOT full content coverage. For files >200 lines: verify file was physically copied into tree, not "inlined" based on grep hits |
| L16 | `skill_view` fails on ambiguous name | Use category-path format `skill_view(name="Category/skill-name")`; if still ambiguous use `read_file` on absolute path |

## shared/ Directory Note

In a Multi-Skill tree, the `shared/` directory exists as a structural placeholder and **may legitimately be empty**. It is only populated when two or more skills have **completely identical** leaf-level instructions (Shared-identical classification per Step B). When all overlapping capabilities are Shared-similar (instructions differ even slightly), each skill gets its own independent leaf node and `shared/` stays empty. An empty `shared/` is correct, not a sign of a missed capability — the GENERATION-REPORT.md Step B section documents the classification decision.
