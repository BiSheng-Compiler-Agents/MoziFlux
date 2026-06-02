# Triton Operator Environment Configuration [LEAF NODE]

Validate and build the Triton operator development environment on Ascend NPU, including
CANN, Python/torch/torch_npu/triton-ascend dependencies, and PATH environment variables.
Use when configuring a Triton operator development environment from scratch or verifying
existing installations.

## MANDATORY: Get Latest Compatibility Requirements First

**Always access the official online documentation first:**
https://github.com/triton-lang/triton-ascend/blob/main/docs/en/installation_guide.md

Obtain the latest:
- Python version requirements
- torch / torch_npu version requirements
- triton-ascend version requirements
- CANN version requirements
- Version compatibility mappings

**The following is for reference only (as of 2026-06-01); official docs prevail:**

| Triton-Ascend Version | torch_npu version | CANN Version | Release Date |
|-----------------------|-------------------|--------------|--------------|
| 3.2.1 | 2.7.1 | 9.0.0 (Recommended) | 2026/06/01 |

---

## Workflow

**Environment checks must be performed in sequence, as each step depends on the success of the previous one.**

### Step 1: Conda Installation and Python Environment

1. Check for existing conda: `conda init bash`
   - If exists: skip to step 5
   - If not: proceed to step 2

2. Determine architecture:
   ```bash
   uname -m
   # aarch64 → wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh
   # x86_64  → wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
   ```

3. Install:
   ```bash
   bash Miniconda3-latest-Linux-x86_64.sh  # or aarch64 variant
   conda init bash
   ```

4. Verify: `conda init bash`

5. Create Python environment (recommended version: 3.11 as of June 2026):
   ```bash
   conda create -n triton python=<version_from_official_docs>
   conda activate triton
   ```

### Step 2: CANN Installation

Download from the [official website](https://www.hiascend.com/cann/download).
Recommended version as of June 2026: 9.0.0.

Install in conda, Ubuntu OS, 950 Ascend product series, x86 CPU architecture:
```bash
# conda's directory requires 755 permissions
conda config --add channels https://repo.huaweicloud.com/ascend/repos/conda/
conda install ascend::cann-toolkit==9.0.0
conda install ascend::cann-950-ops==9.0.0
```

Verification:
```bash
source /home/miniconda3/Ascend/cann/set_env.sh
python3 -c "import acl;print(acl.get_soc_name())"
```

### Step 3: torch and torch_npu Installation

Recommended version as of June 2026: 2.7.1.

Install torch CPU version first:
```bash
pip install torch==2.7.1+cpu --index-url https://download.pytorch.org/whl/cpu
```

Then install torch_npu:
```bash
pip install torch_npu==2.7.1
```

### Step 4: Triton-Ascend Installation

Check the [official page](https://github.com/triton-lang/triton-ascend/tree/main) for the latest version.

Recommended version as of June 2026: 3.2.1:
```bash
pip install triton-ascend==3.2.1 --extra-index-url=https://triton-ascend.osinfra.cn/pypi/simple
```

---

## Verification Checklist

After installation, verify each component:

```bash
# 1. Source CANN environment
source ~/miniconda3/Ascend/cann/bin/setenv.bash

# 2. Check Python
python3 --version

# 3. Check torch
python3 -c "import torch; print(torch.__version__)"

# 4. Check torch_npu
python3 -c "import torch_npu; print(torch_npu.__version__)"

# 5. Check triton-ascend
python3 -c "import triton; print(triton.__version__)"

# 6. Check CANN
python3 -c "import acl; print(acl.get_soc_name())"
```

**CRITICAL**: Always source `~/miniconda3/Ascend/cann/bin/setenv.bash` before running
any Triton-Ascend Python scripts or CANN tools. Without it, `libhccl.so` and other CANN
libs are missing.

---

## Common Installation Issues

**conda not found after install**: Run `source ~/.bashrc` or open a new terminal.

**CANN libs not found at runtime**:
```bash
source ~/miniconda3/Ascend/cann/bin/setenv.bash
# Must be done in every shell session before running Triton kernels
```

**torch_npu version mismatch**: torch and torch_npu versions must match exactly.
Check the official compatibility table.

**triton-ascend build fails**: Ensure CANN is installed and sourced before installing triton-ascend.

**Permission errors on conda directory**: `chmod -R 755 ~/miniconda3/`

---

## Environment Verification Report Format

After completing setup, output a report:

```
Environment Verification Report
================================
Component        | Version     | Status
-----------------|-------------|--------
Python           | 3.11.x      | ✅ PASS
CANN             | 9.0.0       | ✅ PASS
torch            | 2.7.1+cpu   | ✅ PASS
torch_npu        | 2.7.1       | ✅ PASS
triton-ascend    | 3.2.1       | ✅ PASS
setenv sourced   | Yes         | ✅ PASS

Overall: READY for kernel development
```

## Constraints
- Always fetch the latest version requirements from official docs — do not use hardcoded versions without verification
- CANN setenv.bash must be sourced in every shell session before any Triton work
- torch and torch_npu versions must match exactly
- `⚠️ If the versions in the online documentation differ from the reference table, the online documentation must prevail`
