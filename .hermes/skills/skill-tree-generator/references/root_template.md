# ROOT.md Template

This template defines the root-level routing protocol for a skill-tree.

---

## Template

```markdown
# {Skill Name} Routing Protocol [MANDATORY]
Before handling any user task, the following routing flow must be executed:

## Step 1: L1 Routing
Based on the user's **full conversation history + current prompt**, determine the task category.

| Task Category | Routing Target |
|---------------|----------------|
| {category1} | Read `{skills-dir}/{skill-tree}/{module1}/ROUTER.md` |
| {category2} | Read `{skills-dir}/{skill-tree}/{module2}/ROUTER.md` |
| {category3} | Read `{skills-dir}/{skill-tree}/{module3}/SKILL.md` |
| Other/unspecified | Read `{skills-dir}/{skill-tree}/{default}/ROUTER.md` |

## Step 2: Recursive Routing
Continue following the instructions in the ROUTER.md that was read, until encountering a file
marked with `[LEAF NODE]` — that is the final skill.

## Step 3: Execute
Read the leaf node SKILL.md in full, execute the task according to its specification, and print a log that the skill in that directory has been loaded.

## Route Tracing [Optional]

When the user's prompt contains **"routing debug"** / **"debug routing"** / **"routing trace"**, activate route tracing mode:

1. After Step 1 decision, output: `[Route] ROOT → <module> (<matched signal>)`
2. After each ROUTER.md decision, append: `[Route]   → <capability> [LEAF]`
3. Begin execution after reaching the leaf node; no further routing output

**Normal mode** (default): output no routing information, execute directly.

## Important Constraints
- Routing decisions must consider all context constraints established in the conversation
- If the task spans multiple categories, read multiple leaf nodes in parallel
- When the user explicitly specifies a module, route directly to the corresponding subtree
```

---

## Placeholders

| Placeholder | Description | Example |
|-------------|-------------|---------||
| `{Skill Name}` | Name of the skill-tree | `Web Development` |
| `{skills-dir}` | Agent-specific skills directory | `.claude/skills` (Claude Code) / `.agent/skills` (Codex CLI) |
| `{categoryN}` | L1 category matching criteria | `Frontend/UI related` |
| `{skill-tree}` | Directory name of the skill-tree | `web-dev-tree` |
| `{moduleN}` | Module directory name | `frontend` |
| `{default}` | Default fallback module | `general` |

---

## Multi-Skill ROOT.md

Used for cross-domain routing trees that contain multiple skills (Phase 1 selects Skill, Phase 2 selects capability).

```markdown
# {Domain} Routing Protocol [MANDATORY]
Before handling any user task, the following routing flow must be executed:

## Phase 1: Select Skill
Based on the user's **full conversation history + current prompt**, determine which Skill to use.

| User Intent | Keyword Signals | Routing Target |
|-------------|-----------------|----------------|
| {Skill_A capability description} | {Skill_A unique words}, {Skill name A} | Read `./{skill_a}/ROUTER.md` |
| {Skill_B capability description} | {Skill_B unique words}, {Skill name B} | Read `./{skill_b}/ROUTER.md` |
| Shared functionality | {shared signals} | Read `./shared/{capability}/SKILL.md` |
| Cross-skill workflow | workflow, pipeline, batch | Read `./cross-cutting/SKILL.md` |
| Other/unspecified | — | List all Skills and let the user choose |

## Phase 2: Select Capability
Continue routing through the Skill subtree's ROUTER.md until finding a file marked with `[LEAF NODE]`.

## Disambiguation Rules
- Mentions **{Skill name A}** or **{Skill_A unique domain word}** → `{skill_a}/`
- Mentions **{Skill name B}** or **{Skill_B unique domain word}** → `{skill_b}/`
- Mentions only **"{shared keyword}"** with no context → ask the user to choose
- Skill context already established in conversation → do not re-select Skill, go directly into its subtree

### Signal Priority

When the user's prompt contains multiple signals simultaneously, process them in the following priority order:

| Priority | Signal Type | Routing Behavior | Example |
|----------|-------------|------------------|---------|
| **P1 Highest** | Skill name | Route directly, no question | "develop with React" → react |
| **P2** | Unique domain word | Route directly, no question | "undo" → uniquely determines one skill |
| **P3 Lowest** | Cross-domain general word | Only ask when no P1/P2 present | only "create" with no context → ask |

**Priority rule**: P1 > P2 > P3. When P2 and P3 appear together, P2 overrides P3 — no question asked.

## Route Tracing [Optional]

When the user's prompt contains **"routing debug"** / **"debug routing"** / **"routing trace"**, activate route tracing mode:

1. After Phase 1 decision, output: `[Route] ROOT → <skill-name> (P<level>: <matched signal>)`
2. After each ROUTER.md decision, append: `[Route]   → <capability-name> [LEAF]`
3. For cross-skill tasks, output one line per skill
4. Begin execution after reaching the leaf node; no further routing output

**Normal mode** (default): output no routing information, execute directly.

## Important Constraints
- **Must** route before executing; do not skip routing and guess the Skill directly
- **Context first**: when a Skill name has been explicitly mentioned in the conversation, route directly to the corresponding subtree
- **Context accumulation**: routing decisions must consider all context constraints established in the conversation
```

---

## Common Shared Keyword Checklist

After generating, check the following high-frequency shared words one by one and ensure each has a disambiguation rule:
- `create` / `new` — multiple skills may all have a create function
- `export` — multiple skills may all have an export function
- `list` / `info` — almost every skill has a query capability
- `config` / `configure` — may involve different configurations in different skills
- `test` / `testing` — frontend testing vs backend testing vs integration testing
