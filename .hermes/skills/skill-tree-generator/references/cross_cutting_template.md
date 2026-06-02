# Cross-Skill Workflows [LEAF NODE] — Template

> This template is used to generate `cross-cutting/SKILL.md`. cross-cutting is the **workflow combiner** in a Multi-Skill tree, serving two responsibilities:
> 1. **Predefined workflow dictionary** — covers high-frequency cross-skill combination scenarios
> 2. **Custom workflow fallback** — when user intent does not match any predefined workflow, guides the agent to decompose the task itself and route to each skill in turn

## File Structure

The generated `cross-cutting/SKILL.md` must contain the following sections (in order):

```
# Cross-Skill Workflows [LEAF NODE]

## Target Skills
## When to Use
## When NOT to Use
## Predefined Workflows          ← predefined workflow table + bash examples
## Custom Workflow Composition   ← [KEY] fallback mechanism, must be included
## Constraints
```

## Section Generation Rules

### Target Skills
List all skill names in the tree, joined by ` + `.

### When to Use / When NOT to Use
```
## When to Use
- Cross-skill workflows, pipelines, batch processing, cross-domain tasks
- Tasks requiring combining capabilities from different skills
- Output from one skill serves as input to another skill

## When NOT to Use
- Only involves one skill → route to that skill's subtree
- Simple single-step task → use the corresponding leaf node directly
```

### Predefined Workflows
Generate one row for each pair/group of skills that have a natural collaborative relationship:

```markdown
| Workflow | Skills Involved | Description |
|----------|----------------|-------------|
| {workflow name} | {skill_a} + {skill_b} | {brief description of data flow and steps} |
```

Provide an example for each predefined workflow:

```markdown
### Workflow 1: {Name}

**Trigger condition**: {what kind of user request triggers this}

**Steps**:
1. Use {skill_a} to complete {subtask description} → route to `{skill_a}/{module}/SKILL.md`
2. Pass the result of the previous step to {skill_b}
3. Use {skill_b} to complete {subtask description} → route to `{skill_b}/{module}/SKILL.md`
```

### Custom Workflow Composition (must be included)
**This is the core capability of cross-cutting and must not be omitted.**

```markdown
## Custom Workflow Composition (fallback mechanism when no predefined match)

When user intent involves multiple skills but is not in the predefined workflow table above, follow this process to compose:

### Step 1: Identify Involved Skills
Extract each subtask and its corresponding skill from the user intent (refer to the keyword signals in ROOT.md Phase 1).

### Step 2: Determine Execution Order
Arrange by data flow: which skill's output is the next skill's input.
- Clear sequential dependency → execute serially
- No dependency → execute in parallel

### Step 3: Route Each Skill to a Specific Leaf Node
For each subtask, go back to the corresponding skill's ROUTER.md → leaf SKILL.md to find the precise execution instructions. Do not guess instructions; you must find them by traversing the routing tree.

### Step 4: Execute in Series
In the order determined in Step 2, read and execute each leaf node's instructions in turn. The output of the previous step is the input to the next step.
```

When generating, **provide a concrete example using the actual skills in the tree**, demonstrating the complete decompose → route → execute process.

## Checklist

- [ ] Target Skills lists all skills in the tree (none missing)
- [ ] Each row in the predefined workflow table involves ≥ 2 skills
- [ ] Each predefined workflow has a corresponding steps example
- [ ] **Custom Workflow Composition section exists and contains 4 steps**
- [ ] Custom workflow example uses skills that actually exist in the tree
