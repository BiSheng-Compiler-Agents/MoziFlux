# NexusOpt

A research project for optimizing Triton kernels targeting Ascend NPUs using
agentic systems. Contains kernel implementations, benchmarks, cannsim
simulation infrastructure, and an optimization pattern knowledge base.

---

## Environment Setup

### 1. Create the conda environment

```bash
conda env create -n nexusopt -f environment.yml
```

If that fails, install the required packages manually in order:

**Step 1 — CANN 9.0.0**

Download from the [CANN download page](https://www.hiascend.com/cann/download?versionId=717&ids=d806%2Ch0502%2Ch0602%2Ch0701) and follow the installer instructions.

After installation, always source the environment before using any CANN or
Triton-Ascend tools:
```bash
source ~/miniconda3/Ascend/cann/bin/setenv.bash
```

**Step 2 — Triton-Ascend 3.2.1**
```bash
pip install triton-ascend==3.2.1 --extra-index-url=https://triton-ascend.osinfra.cn/pypi/simple
```

**Step 3 — PyTorch (CPU)**
```bash
pip install torch==2.7.1+cpu --index-url https://download.pytorch.org/whl/cpu
```

**Step 4 — torch-npu**
```bash
pip install torch_npu==2.7.1
```

## Hermes Agent Integration

This project integrates with [Hermes Agent](https://github.com/NousResearch/hermes-agent)
for agentic kernel optimization workflows. Skills and plugins specific to this
project live outside the default Hermes installation directory.

### Using project-specific skills

Skills located outside `~/.hermes/skills/` (the default) can be registered via
`external_dirs` in `~/.hermes/config.yaml`:

```yaml
skills:
  external_dirs: [
    - /path/to/NexusOpt/skills
  ]
```

- Paths support `~` and `${ENV_VAR}` expansion
- External skill directories are non-destructive — Hermes reads from them
  but never writes to them automatically
- Skills in external dirs appear in `hermes skills list` and `/skill` commands
- If a skill name conflicts, later entries in the list take precedence over
  earlier ones (user `~/.hermes/skills/` always takes precedence over external)

### Using project-specific plugins

Hermes supports **project-local plugins** placed in a `.hermes/plugins/`
directory relative to the working directory where Hermes is launched.

**Step 1** — Create the plugin directory structure in your project:

```
NexusOpt/
└── .hermes/
    └── plugins/
        └── my-plugin/
            ├── plugin.yaml
            └── __init__.py
```

**Step 2** — Enable project plugin loading by setting the environment variable:

```bash
export HERMES_ENABLE_PROJECT_PLUGINS=true
```

Or add it to `~/.hermes/.env`:
```
HERMES_ENABLE_PROJECT_PLUGINS=true
```

Make sure to add the toolset names registered in the plugins in different platforms in the config file as well. 
For example, add `triton_ascend` toolset which contains RAG tools, and cannsim tool to the `cli` platform in the config file of hermes to make them available when using Hermes from CLI. 
Without this, the tools won't be available to the agent in that specific platform. 
Also, put the plugin names in the `plugins.enabled` section in the yaml file. Without this, the plugins themselves and the tools they provide won't detected by Hermes.

**Step 3** — Launch Hermes from the project root (so `./.hermes/plugins/`
resolves correctly):

```bash
cd /path/to/NexusOpt
hermes
```

**Plugin precedence** (highest to lowest):
1. Project plugins (`./.hermes/plugins/`) — only when `HERMES_ENABLE_PROJECT_PLUGINS=true`
2. User plugins (`~/.hermes/plugins/`) — always loaded
3. Bundled plugins (shipped with Hermes)

A project plugin with the same name as a user or bundled plugin overrides it.

> **Note:** Project plugins must also be listed under `plugins.enabled` in
> `~/.hermes/config.yaml` to be activated, just like any other plugin.

---

