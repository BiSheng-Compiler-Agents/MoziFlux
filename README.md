# MoziFlux

MoziFlux is an agentic system for optimizing Triton kernels targeting Ascend
NPUs. This repository contains kernel implementations, benchmarks,
[cannsim](https://gitcode.com/cann/ops-transformer/blob/master/docs/zh/debug/cann_sim.md)
simulation infrastructure, and an optimization pattern knowledge base.

MoziFlux integrates with [Hermes Agent](https://github.com/NousResearch/hermes-agent)
to drive its agentic kernel optimization workflows. Skills and plugins specific
to this project live outside the default Hermes installation directory.

MoziFlux is named after [Mozi](https://en.wikipedia.org/wiki/Mozi), the Chinese
philosopher, logician, and engineer.

## Environment Setup

### 1. Create the Conda environment

Currently, MoziFlux requires Linux on an x86_64 machine, preferably Ubuntu 22.04
or newer. If your system already has [Conda](https://anaconda.org/) set up, you
can try this simple command:

```bash
conda env create -n moziflux -f environment.yml
```

If that fails, install the required packages manually in the following order:

#### 1.1 CANN 9.0.0

Download from the [CANN download page](https://www.hiascend.com/cann/download?versionId=717&ids=d806%2Ch0502%2Ch0602%2Ch0701)
and follow the installer instructions.

After installation, always source the environment before using any CANN or
Triton-Ascend tools:
```bash
source ~/miniconda3/Ascend/cann/bin/setenv.bash
```

#### 1.2 Ascend 3.2.1

```bash
pip install triton-ascend==3.2.1 --extra-index-url=https://triton-ascend.osinfra.cn/pypi/simple
```

#### 1.3 PyTorch (CPU)

```bash
pip install torch==2.7.1+cpu --index-url https://download.pytorch.org/whl/cpu
```

#### 1.4 torch-npu

```bash
pip install torch_npu==2.7.1
```

### 2. Install Hermes Agent

Follow the [Hermes Agent Quick Install](https://github.com/NousResearch/hermes-agent#quick-install)
instructions:

```bash
curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
source ~/.bashrc
```

### 3. Set up MoziFlux skills for Hermes Agent

Skills located outside `~/.hermes/skills/` (the default directory where the
agent finds its skills) can be registered via `external_dirs` in
`~/.hermes/config.yaml`:

```yaml
skills:
  external_dirs: [
    - /path/to/MoziFlux/skills
  ]
```

- Path names support `~` and `${ENV_VAR}` expansion.
- External skill directories are non-destructive&mdash;Hermes reads from them
  but never writes to them automatically.
- Skills in external directories appear in `hermes skills list` and `/skill`
  commands.
- If there is a skill name conflict, later entries in the list take precedence
  over earlier ones; the user's `~/.hermes/skills/` always takes precedence
  over external directories.

### 4. Set up project-local plugins

Hermes supports **project-local plugins** placed in a `.hermes/plugins/`
directory relative to the working directory where Hermes is launched.

#### 4.1 Create the plugin directory structure in your project

```
MoziFlux/
└── .hermes/
    └── plugins/
        └── my-plugin/
            ├── plugin.yaml
            └── __init__.py
```

#### 4.2 Enable Project Plugins

To tell Hermes to look for project plugins, set this environment variable:

```bash
export HERMES_ENABLE_PROJECT_PLUGINS=true
```

Or add it to `~/.hermes/.env`:

```
HERMES_ENABLE_PROJECT_PLUGINS=true
```

In the Hermes configuration file (`~/.hermes/config.yaml`), list the project
plugins under `plugins.enabled`. Without this, the plugins (and the tools they
provide) will not be detected by Hermes.

Most plugins register one or more **toolsets**. To allow Hermes to use a
toolset on a given platform (user interface), you must also add the name of the
toolset to the platform in the Hermes configuration file. For example, add
`triton_ascend` toolset (which contains RAG tools and the `cannsim` tool) to
the `cli` platform in the configuration file, if you plan to use MoziFlux via
the Hermes CLI. If the name of the toolset is not configured for a platform you
selected, the tools will not be available to the agent on that platform.

Plugins are loaded in this order (highest precedence to lowest):

1. Project plugins (`./.hermes/plugins/`): only when `HERMES_ENABLE_PROJECT_PLUGINS=true`.
2. User plugins (`~/.hermes/plugins/`): always loaded.
3. Bundled plugins (shipped with Hermes).

A project plugin with the same name as a user or bundled plugin overrides the
latter.

For more information on configuring Hermes, see
[the official documentation](https://hermes-agent.nousresearch.com/docs/user-guide/features/plugins).

#### 4.3 Launch Hermes from the MoziFlux project root

Remember, the project root is where the relative path `./.hermes/plugins/` can
be located. You can now launch Hermes and start using MoziFlux.

```bash
cd /path/to/MoziFlux
hermes
```
