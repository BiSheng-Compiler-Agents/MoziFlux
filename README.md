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

## Table of Contents

- [Quick Start](#quick-start)
- [Environment Setup](#environment-setup)
- [Plugins \& Environment Variables](#plugins--environment-variables)
- [Running with Docker (Recommended)](#running-with-docker-recommended)
- [Usage Inside the Container](#usage-inside-the-container)
  - [Hermes Agent CLI](#hermes-agent-cli)
  - [Python API / Scripts](#python-api--scripts)
- [Launch Hermes from the MoziFlux project root](#launch-hermes-from-the-moziflux-project-root)

---

## Quick Start

The fastest way to get running is via Docker — it bundles CANN, Triton-Ascend,
Hermes Agent, and all dependencies in a sandboxed container:

```bash
cd docker
bash run_container.sh
```

This builds the image, starts the container, and drops you into a Hermes chat
session. See [Running with Docker](#running-with-docker-recommended)
for full details.

---

## Environment Setup

Only needed if you want to run Hermes directly on your host (without Docker).

### 1. Create the Conda environment

Currently, MoziFlux requires Linux on an x86_64 machine, preferably Ubuntu 22.04
or newer. If your system already has [Conda](https://anaconda.org/) set up, you
can try this simple command:

```bash
conda env create -n moziflux -f docker/environment.yml
```

If that fails, install the required packages manually in the following order:

#### 1.1. Install CANN 9.0.0

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

## Plugins & Environment Variables

### 1. MoziFlux skills for Hermes Agent

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
- Skills in external directories appear in `hermes skills list` and `/skill`
  commands.
- If there is a skill name conflict, later entries in the list take precedence
  over earlier ones; the user's `~/.hermes/skills/` always takes precedence
  over external directories.

### 2. Project-local plugins for Hermes Agent

Hermes supports **project-local plugins** placed in a `.hermes/plugins/`
directory relative to the working directory where Hermes is launched.
Plugins have the following structure:

```
MoziFlux/
└── .hermes/
    └── plugins/
        └── my-plugin/
            ├── plugin.yaml
            └── __init__.py
```

### 3. Hermes config file

Copy `config_example.yaml` to `~/.hermes/config.yaml` and customize:

```bash
cp config_example.yaml ~/.hermes/config.yaml
```

The example already includes:
- `skills.external_dirs` pointing to `/opt/moziflux/.hermes/skills`
(project skills)
- `plugins.enabled` listing all 6 project plugins

Make sure these sections are present in your config:

```yaml
skills: # for adding local skills
  external_dirs:
  - /opt/moziflux/.hermes/skills

plugins: # for adding local plugins
  enabled:
  - cannsim-local
  - cannsim-remote
  - kernel-episodes
  - kernel-sandbox
  - phoenix-tracer
  - remote-verify
```

### 4. Environment variables

Copy `.env.example` to `~/.hermes/.env` and fill in your values:

```bash
cp .env.example ~/.hermes/.env
```

Key variables (see `.env.example` for full list):

```bash
# ── Required for LLM access ──
OPENROUTER_API_KEY=YOUR_API_KEY

# ── cannsim-local plugin ──
CANNSIM_SETENV_PATH=/opt/miniconda3/Ascend/cann/set_env.sh
CONDA_BIN=/opt/miniconda3/bin/conda
CONDA_ENV=compilerclaw

# ── remote-verify plugin (optional) ──
REMOTE_VERIFY_HOST=HOSTNAME_FOR_REMOTE_A5_MACHINE
REMOTE_VERIFY_USER=USERNAME_FOR_REMOTE_A5_MACHINE
REMOTE_VERIFY_PASS=PASS_FOR_REMOTE_A5_MACHINE

# ── Required for project plugins to load ──
HERMES_ENABLE_PROJECT_PLUGINS=true
```

> **Docker note:** When running inside the container,
> and `HERMES_ENABLE_PROJECT_PLUGINS` are pre-configured in the Dockerfile.
> You only need to set `OPENROUTER_API_KEY` and optionally `REMOTE_VERIFY_*`.

---

## Running with Docker (Recommended)

The Docker container provides a sandboxed environment with everything
pre-installed: CANN 9.0.0, Triton-Ascend 3.2.1, PyTorch, and Hermes Agent.
All CLI and Python API usage happens inside this container.

### Build and launch

```bash
cd docker
bash run_container.sh          # Build + start + enter Hermes chat
bash run_container.sh shell    # Build + start + enter bash shell
bash run_container.sh exec     # Build + start, print attach commands
```

### Resource limits (optional)

```bash
bash run_container.sh hermes 4 8g    # 4 CPUs, 8GB memory
```

### How it works

- Builds from `nousresearch/hermes-agent:latest` base image
- Installs CANN, Miniconda, Triton-Ascend, and all dependencies
- Mounts `$HOME/.hermes` (config + credentials) and the project directory
- Entrypoint remaps UID/GID to match the host user (so files are writable)
- Runs `sleep infinity` as main process — use `docker exec` to interact

---

## Usage Inside the Container

Once the container is running (`run_container.sh` launches it automatically),
use `docker exec` to run commands inside it. All commands below run as the
`hermes` user inside the container.

### Hermes Agent CLI

Start an interactive Hermes chat session:

```bash
docker exec -it -u hermes moziflux hermes
```

Browse past sessions:

```bash
docker exec -it -u hermes moziflux hermes sessions browse
```

Other Hermes CLI commands work the same way — just prefix with
`docker exec -it -u hermes moziflux`.

### Python API / Scripts

Run any Python script inside the container:

```bash
# Generate aggregate performance report
docker exec -u hermes moziflux bash -c 'python generate_report.py'

# Run the kernel optimization orchestrator
docker exec -u hermes moziflux bash -c 'python optimize_kernels.py'

# Run a single kernel profile
docker exec -u hermes moziflux bash -c 'cd datasets/KernelBench_Triton/l1_19_ReLU && python profile_kernels.py --test --bench'
```

Or get a shell first, then run commands interactively:

```bash
docker exec -it -u hermes moziflux /bin/bash
python generate_report.py
python optimize_kernels.py --dry-run
```

### Output files

Generated reports appear in `reports/` (created at runtime):
- `reports/per_kernel/<name>.png` — one line plot per kernel
- `reports/aggregate_runtime_boxplot.png` — runtime distribution per method
- `reports/aggregate_geomean_speedup_vs_*.png` — geomean speedup bar charts
- `reports/all_runtimes.csv` — raw data dump

---

## Launch Hermes from the MoziFlux project root

Remember, the project root is where the relative path `./.hermes/plugins/` can
be located. You can now launch Hermes and start using MoziFlux.

```bash
cd /path/to/MoziFlux
hermes
```
