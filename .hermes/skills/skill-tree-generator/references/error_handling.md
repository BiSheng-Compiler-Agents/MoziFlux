# Error Handling & Reference File Processing (Unified Specification)

The rules in this section apply to all generation operations in Mode 1/2/3.

## Error Severity & Handling Strategy

| Level | Trigger Condition | Handling |
|-------|------------------|----------|
| **Fatal** | Source skill file does not exist (path invalid or no SKILL.md) | Immediately report error and stop; list all missing skill paths |
| **Fatal** | Source skill file exists but content is empty (0 bytes or no valid instruction content) | Immediately report error and stop |
| **Degraded** | Source skill exists, but the content for a particular sub-capability/sub-section is not obtainable (e.g. a parsed sub-module has no corresponding section in the original skill) | Generate a fallback leaf node marked with `[AUTO-GENERATED FALLBACK - NO SOURCE CONTENT]`, containing minimal executable instructions inferred from context. **Prohibited**: generating stubs containing only title + summary + external link |
| **Warning** | An external file referenced in the source skill does not exist in the source skill package | Delete that reference entry; replace with self-contained directly executable instructions (e.g. use Grep/Read instead of a script call); continue generation |

**Fatal vs Degraded boundary**: Source skill **as a whole** is unavailable (file does not exist / entirely empty) → Fatal. Source skill **as a whole** is available but **local content** is missing → Degraded.

**Degraded ratio threshold**: If the number of Degraded leaf nodes exceeds 30% of the total leaf node count, issue a warning and list the affected nodes for user confirmation before continuing.

## Atomicity Guarantees

- Complete all content generation (in memory or a temporary location) before creating/modifying the tree directory structure
- When copying large directories (>5 files or >50KB), prefer platform-native commands: Unix/macOS use `cp -a` (`cp -r` as fallback only), Windows use `robocopy`. Prohibited: copying file by file with the Write tool; `xcopy /E /I` is only a last fallback when `robocopy` is unavailable
- Large directory copies must use a staging strategy: first copy to a brand-new staging directory inside the tree or a temporary location; after successful verification, replace/move to the final target directory; never overwrite the final target directory directly
- On Windows, `robocopy` return codes `< 8` are success, `>= 8` are failure; do not treat robocopy as "non-zero means failure"
- If a filesystem operation (copying a large directory, replacing a target directory, etc.) fails, report the error and clean up the staging/temporary files generated in this run; do not leave a partial tree

- Before replacing an existing target directory, avoid mixing in old files: generate complete content in a brand-new staging directory, then replace the target directory after verification, or explicitly delete old files in the target that are not in this run's output manifest

---

## Reference File Processing Flow

**Definition of "source skill package"**: The entire directory tree rooted at the directory containing the source skill's SKILL.md. For example, all files under `.claude/skills/my-skill/` (including subdirectories).

### Step R1: Inventory External References

For each external file/directory referenced in the source skill content:

1. Resolve relative paths relative to the directory containing the source skill's SKILL.md
2. Use Glob to confirm whether the file/directory exists
3. For each found file/directory, resolve the real (absolute/canonical) path, confirm it is still within the source skill package root directory; prohibited: introducing content outside the source skill package via `..`, symlinks, junctions, or absolute paths
4. Exists and real path is within the source skill package → continue to Step R2
5. Does not exist or real path escapes outside the source skill package → mark as "unavailable"

### Step R2: Decide Handling Strategy

| Reference Type | Quantitative Criteria | Handling |
|---------------|-----------------------|----------|
| Short file | ≤ 200 lines **and** ≤ 10KB | Inline complete content into leaf node |
| Medium file set | 2–5 files, each > 200 lines | Evaluate individually: prefer inlining; may copy into tree when total lines > 800 |
| Large file set | > 5 files **or** total size > 50KB | **Use staging + platform-native command to copy the entire directory** into the corresponding position inside the tree (e.g. `{skill}/docs/`): Unix/macOS prefer `cp -a`, Windows prefer `robocopy`, `xcopy /E /I` only as last fallback. **Prohibited: copying file by file with the Write tool**. Leaf nodes may use tree-internal relative paths (`../docs/` or `./docs/`) |
| Unavailable | File does not exist in source skill package | Delete the reference entry; replace with self-contained directly executable instructions (e.g. use Grep/Read instead of a script call, use inline data instead of an external JSON file) |

**Mixed scenarios**: If the source skill contains both short file references and large file set references, handle each according to its own criteria.

**Copy filter rules**: When copying a directory, by default exclude: `.git/`, `.svn/`, `.hg/`, `node_modules/`, `dist/`, `build/`, `coverage/`, cache directories, temp files, `.DS_Store`, `Thumbs.db`, `.env`, `.env.*`, secret/certificate files (e.g. `*.pem`, `*.key`). Only explicitly excluded types may be included if the source skill explicitly requires it and they contain no sensitive information.

**Copy verification rules**: After the staging directory is generated, verify at minimum the file count, total size, existence of key entry files, and that target reference paths are relatively accessible from leaf nodes. Verification failure must clean up the staging directory and report an error; generation must not continue.

### Step R3: Post-Generation Cleanup

After all leaf nodes are generated, execute the following scan and cleanup:

**R3.1 Scan for residual references** — Grep all leaf nodes for the following patterns (case-insensitive):

| Category | Grep Patterns |
|----------|---------------|
| Chinese reference sections | `参考文档索引`, `参考资料`, `相关文档`, `扩展阅读` |
| English reference sections | `Reference Documents`, `References`, `See Also`, `Further Reading`, `Related Documents`, `External References` |
| External pointer phrases | `For more details, see`, `Additional resources`, `Refer to`, `See {path}`, `Read {path}`, `Full Instructions: Read`, `Read the skill at` |
| Tree-external absolute paths | `\.claude/skills/(?!.*-tree/)` (matches `.claude/skills/xxx/` but not `.claude/skills/xxx-tree/`) |

**R3.2 Three-step handling per reference entry**:

- File already copied into tree → replace path with tree-internal relative path
- Content already inlined into leaf node → delete this reference entry (content is already in the leaf node)
- File does not exist in source skill package → delete the entire reference section

**R3.3 External path replacement** — Grep all leaf nodes for absolute paths pointing outside the tree; replace all with tree-internal relative paths or delete.

### Step R4: Immediate Validation

Before entering Step 6 / Step D / final output for each mode, the following checks **must** be executed (do not wait for the final validation phase):

1. **Stub detection**: Grep all newly generated leaf nodes; confirm they do not contain the stub/external pointer patterns listed in Step R3.1
2. **External path detection**: Grep all newly generated leaf nodes; confirm they do not contain absolute paths pointing outside the tree
3. If a match is found → fix immediately; re-check after fixing
4. Only continue subsequent steps after passing

---

## Self-Containment Rule [MANDATORY]

The skill tree must be self-contained as a whole, completely independent from the source skill package and any paths outside the tree. Leaf nodes should prefer inlining content required for execution; when a reference belongs to a large file set as defined by Step R2, a leaf node may reference files already copied into the same tree using tree-internal relative paths. Strictly prohibited: creating stub files containing only a summary and a pointer to a file outside the tree. An agent loading only the tree must have everything needed to execute. This includes:

- Feature dictionaries with Min/Max ranges and conversion coefficients
- API endpoint specifications, including endpoint addresses, parameters, and response formats
- Enum values, lookup tables, classification codes
- Executable workflow steps, code examples, and usage patterns
- Any data contained in or referenced by the original skill

If the original skill references external files, decide whether to inline or copy into the tree according to Step R2. Short files must be inlined into the corresponding leaf nodes; large file sets may be copied into the tree and referenced using tree-internal relative paths. References of the type "see file X" pointing to the source skill package or other locations outside the tree must not be retained.
