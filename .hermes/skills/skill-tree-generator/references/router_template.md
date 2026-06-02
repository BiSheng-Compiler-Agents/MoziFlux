# ROUTER.md Template

This template defines the routing logic for non-leaf nodes in a skill-tree.

---

## Template

```markdown
# {Module Name} Router [Ln]

You have reached the {module} subtree. Determine the next step based on the current task:

| Condition | Next Hop |
|-----------|----------|
| {condition1} | Read `./{path1}/ROUTER.md` |
| {condition2} | Read `./{path2}/SKILL.md` |
| {condition3} | Read `./{path3}/SKILL.md` |
| Other/unspecified | Read `./{default}/SKILL.md` |

Note: If the user has explicitly stated {context_hint} in the current conversation, prioritize conversation context.

**Route tracing**: When trace mode is active, output `[Route]   → <matched capability> [LEAF]`. If multiple match, output one line per match.
```

---

## Level Markers

- `[L1]` - First level below ROOT
- `[L2]` - Second level
- `[L3]` - Third level
- etc.

---

## Routing Conditions Guidelines

### Condition Types

1. **Keyword Matching**
   ```
   | Involves React/Vue/DOM | Read `./frontend/ROUTER.md` |
   ```

2. **Task Pattern**
   ```
   | Create/new/generate | Read `./create/SKILL.md` |
   | Modify/edit/update  | Read `./edit/SKILL.md` |
   | Delete/remove       | Read `./delete/SKILL.md` |
   ```

3. **Domain Terminology**
   ```
   | API/REST/GraphQL        | Read `./api/SKILL.md` |
   | Database/SQL/NoSQL      | Read `./database/SKILL.md` |
   ```

4. **File Type**
   ```
   | .tsx/.jsx files         | Read `./react/SKILL.md` |
   | .css/.scss/.less        | Read `./css/SKILL.md` |
   ```

### Condition Composition

- **Single keyword**: `React`
- **Multiple keywords (OR)**: `React/Vue/DOM`
- **Pattern**: `create/new/generate`
- **Negation**: Avoid using; prefer positive conditions

---

## Path References

| Reference Type | Syntax | Example |
|---------------|--------|---------||
| Sub-router | `./{dir}/ROUTER.md` | `./frontend/ROUTER.md` |
| Leaf skill | `./{dir}/SKILL.md` | `./react/SKILL.md` |
| Parent level | `../ROUTER.md` | (rarely used) |

---

## Context Hints

The `{context_hint}` should guide the router to consider relevant conversation context:

| Module Type | Context Hint Example |
|-------------|---------------------|
| Frontend | `tech stack (React/Vue/etc.)` |
| Backend | `backend language/framework` |
| Database | `database type` |
| DevOps | `deployment environment` |
| Writing | `document type/target audience` |

---

## Complete Example

```markdown
# Frontend Router [L2]

You have reached the frontend subtree. Determine the next step based on the current task:

| Condition | Next Hop |
|-----------|----------|
| Involves React/JSX/components/hooks | Read `./react/SKILL.md` |
| Involves Vue/Composition API         | Read `./vue/SKILL.md` |
| Involves CSS/styles/animations       | Read `./css/SKILL.md` |
| Involves TypeScript/types            | Read `./typescript/SKILL.md` |
| Involves testing/Jest/Cypress        | Read `./testing/SKILL.md` |
| Other/general frontend               | Read `./general/SKILL.md` |

Note: If the user has explicitly stated the frontend tech stack in the current conversation, prioritize conversation context.
```

---

## Best Practices

1. **Mutual Exclusivity**: Conditions should not overlap significantly
2. **Coverage**: Every possible input should match at least one condition
3. **Specificity**: Order conditions from most specific to most general
4. **Brevity**: Keep conditions concise but unambiguous
5. **Context First**: Always mention conversation context priority
