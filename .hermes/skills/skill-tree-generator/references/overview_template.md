# SKILL-TREE.md Overview Template

> This template is used to generate the `SKILL-TREE.md` overview file in the tree directory.

---

## Single-Skill Mode

````markdown
# {Skill Name} Tree Overview

## Overview
This skill-tree organizes {N} capabilities into a hierarchical routing tree.

## Routing Principle
```
User intent → ROOT.md (L1 module) → ROUTER.md (L2 sub-module) → LEAF SKILL.md (specific capability)
```

## Directory Structure
```
{skill-name}-tree/
├── ROOT.md                          # L1: routing for {N} modules
├── SKILL-TREE.md                    # this file
│
├── {module1}/                       # {module1_display}
│   ├── ROUTER.md                    # disambiguation for {sub_count} sub-modules
│   └── {sub1}/SKILL.md              # → {capabilities}
│
└── {module2}/                       # {module2_display}
    └── SKILL.md                     # → {capabilities}
```

## Capability → Leaf Node Mapping Table

| Capability | Leaf Node Path |
|------------|----------------|
| `{capability_1}` | `{module}/{sub}/SKILL.md` |
| `{capability_2}` | `{module}/{sub}/SKILL.md` |

## Guide for Adding Capabilities
1. Determine the owning module (L1 category)
2. If the module already has a ROUTER.md, add a new row to the routing table
3. If the module has no ROUTER.md, create a new module directory + ROUTER.md
4. Create or update the leaf node SKILL.md
5. Update the mapping table in this file
````

---

## Multi-Skill Mode

````markdown
# {Domain} Tree Overview

## Overview
This skill-tree organizes {N} capabilities into a cross-skill hierarchical routing tree, covering {M} skills.

## Routing Principle
```
User intent → ROOT.md (Phase 1: select Skill) → ROUTER.md (Phase 2: select capability) → LEAF SKILL.md
```

## Directory Structure
```
{domain}-tree/
├── ROOT.md                          # Phase 1: select Skill
├── SKILL-TREE.md                    # this file
│
├── {skill_a}/                       # {Skill_A} subtree
│   ├── ROUTER.md                    # Phase 2: select capability
│   └── {capability}/SKILL.md
│
├── {skill_b}/                       # {Skill_B} subtree
│   ├── ROUTER.md
│   └── {capability}/SKILL.md
│
├── shared/                          # Shared capabilities
│   └── {capability}/SKILL.md        # annotated: applicable to skill_a, skill_b
│
└── cross-cutting/
    └── SKILL.md                     # cross-skill workflows
```

## Capability → Leaf Node Mapping Table

| Skill | Capability | Leaf Node Path |
|-------|------------|----------------|
| `skill-a` | {capability} | `skill-a/{module}/SKILL.md` |
| `skill-b` | {capability} | `skill-b/{module}/SKILL.md` |
| `skill-a, skill-b` | {shared capability} | `shared/{capability}/SKILL.md` |

## Skill Coverage Statistics

| Skill | Capability Count | Leaf Node Count |
|-------|-----------------|-----------------|
| `{skill_a}` | {N} | {M} |
| `{skill_b}` | {N} | {M} |
| shared | {N} | {M} |
| cross-cutting | {N} workflows | 1 |
| **Total** | **{total}** | **{total}** |

## Guide for Adding a Skill
1. Create `{new-skill}/ROUTER.md` + leaf nodes
2. Update ROOT.md Phase 1 routing table + disambiguation rules
3. Update cross-cutting/SKILL.md (workflows + dependencies)
4. Update the mapping table and coverage statistics in this file
````
