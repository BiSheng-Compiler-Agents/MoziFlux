---
name: triton-operator-env-config
description: Validate and build the Triton operator development environment on Ascend NPU, including CANN, Python/torch/torch_npu/triton-ascend dependencies, and PATH environment variables. Use when configuring a Triton operator development environment, checking CANN/torch/triton-ascend installations, or verifying environment availability.
---

# Triton Operator Development Environment Configuration
**MANDATORY - READ ENTIRE PAGE**: Access and fully read the official online documentation at https://github.com/triton-lang/triton-ascend/blob/main/docs/en/installation_guide.md to obtain the latest:
- Python version requirements
- torch / torch_npu version requirements
- triton-ascend version requirements
- CANN version requirements
- Version compatibility mappings between components

**Use the version requirements from the official documentation as the standard, updating version numbers in subsequent steps accordingly.**

**The following version compatibility information is for reference only (as of 2026-06-01); refer to the official online documentation for the definitive information:**

| Triton-Ascend Version | torch_npu version | CANN Version | Release Date |
|-----------------------|-------------------|--------------|--------------|
| 3.2.1 | 2.7.1 | 9.0.0 (Recommended) | 2026/06/01 |


⚠️ If the versions in the online documentation differ from the table above, **the online documentation must prevail**.


**Environment checks must be performed in sequence, as each step depends on the success of the previous one.**

## Prerequisite: Get Latest Compatibility Requirements (MANDATORY)
### Conda
1. For conda installation, first check for an existing conda environment using `conda init bash`. If it exists, skip to step 5. If not, proceed to step 2 to install.
2. Run `uname -m` to confirm the current system architecture.
   - If the system architecture is aarch64, execute: `wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh`
   - If the system architecture is x86_64, execute: `wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh`
3. Execute the installation script: `bash Miniconda3-latest-Linux-x86_64.sh` or `bash Miniconda3-latest-Linux-aarch64.sh`, follow the prompts and then activate the miniconda environment.
4. Check if miniconda was successfully installed: `conda init bash`
5. Create a Python environment by executing: `conda create -n triton python=<version required by official docs>`
6. Activate the Python environment by executing: `conda activate triton`

The recommended python version as of June 2026 is 3.11.

### CANN

You can download the CANN package from the [official website](https://www.hiascend.com/cann/download).

The recommended version as of June 2026 is 9.0.0.
Intallation in conda, Ubuntu OS, 950 Ascend product series, and x86 cpu architecture:
```bash
#conda's directory requires 755 permissions
conda config --add channels https://repo.huaweicloud.com/ascend/repos/conda/
conda install ascend::cann-toolkit==9.0.0
conda install ascend::cann-950-ops==9.0.0
```

Verification:
```bash
source /home/miniconda3/Ascend/cann/set_env.sh

python3 -c "import acl;print(acl.get_soc_name())"
```

### torch_npu
The recommended version as of June 2026 is 2.7.1.
You should first install the torch cpu version:
```bash
pip install torch==2.7.1+cpu --index-url https://download.pytorch.org/whl/cpu
```

Finally, install the torch_npu:
```bash
pip install torch_npu==2.7.1
```

### Triton-Ascend
You can download the latest triton-ascend version using pip or other methods. Check out the [official page](https://github.com/triton-lang/triton-ascend/tree/main) for more. **As compatibility versions may update, you must first obtain the latest official documentation.**

The recommended version as of June 2026 is 3.2.1:
```bash
pip install triton-ascend==3.2.1 --extra-index-url=https://triton-ascend.osinfra.cn/pypi/simple
```
