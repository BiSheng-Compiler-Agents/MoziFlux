# =============================================================================
# NexusOpt Dockerfile
# Base: Hermes Agent official image + NexusOpt conda environment
# =============================================================================

FROM nousresearch/hermes-agent:latest

USER root

# -----------------------------------------------------------------------------
# 1. Install system build tools (cmake, make, g++)
#    Needed to compile the C++ host launcher for cannsim
# -----------------------------------------------------------------------------
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        wget bzip2 cmake make g++ gcc && \
    rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# 2. Install Miniconda
# -----------------------------------------------------------------------------
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/miniconda3 && \
    rm /tmp/miniconda.sh

ENV PATH="/opt/miniconda3/bin:${PATH}"

# -----------------------------------------------------------------------------
# 3. CANN 9.0.0
#    Download from: https://www.hiascend.com/cann/download?versionId=717
# -----------------------------------------------------------------------------
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

RUN conda config --add channels https://repo.huaweicloud.com/ascend/repos/conda/  && \
    conda install ascend::cann-toolkit==9.0.0 && \
    conda install ascend::cann-950-ops==9.0.0

# -----------------------------------------------------------------------------
# 4. Create the compilerclaw conda environment from environment.yml
# -----------------------------------------------------------------------------
COPY environment.yml /tmp/environment.yml

RUN conda create -n compilerclaw python=3.11 && \
    conda clean -afy && \
    rm /tmp/environment.yml

# -----------------------------------------------------------------------------
# 5. Install triton-ascend and torch-npu (need special index URLs)
#    Must run AFTER conda env create so the compilerclaw env exists.
# -----------------------------------------------------------------------------
RUN /opt/miniconda3/envs/compilerclaw/bin/pip install \
    triton-ascend==3.2.1 \
    --extra-index-url=https://triton-ascend.osinfra.cn/pypi/simple && \
    /opt/miniconda3/envs/compilerclaw/bin/pip install \
    torch==2.7.1+cpu \
    --index-url=https://download.pytorch.org/whl/cpu && \
    /opt/miniconda3/envs/compilerclaw/bin/pip install \
    torch_npu==2.7.1

# -----------------------------------------------------------------------------
# 6. Install entrypoint script to fix UID/GID mismatch at container start
# -----------------------------------------------------------------------------
COPY entrypoint.sh /usr/local/bin/nexusopt-entrypoint.sh
RUN chmod +x /usr/local/bin/nexusopt-entrypoint.sh

# -----------------------------------------------------------------------------
# 7. Activate compilerclaw env system-wide
# -----------------------------------------------------------------------------
ENV PATH="/opt/miniconda3/envs/compilerclaw/bin:/opt/miniconda3/bin:/opt/hermes/bin:/opt/hermes/.venv/bin:/opt/data/.local/bin:${PATH}"
ENV CONDA_DEFAULT_ENV=compilerclaw
ENV CONDA_PREFIX=/opt/miniconda3/envs/compilerclaw

# -----------------------------------------------------------------------------
# 8. Enable project plugins
# -----------------------------------------------------------------------------
ENV HERMES_ENABLE_PROJECT_PLUGINS=true

# -----------------------------------------------------------------------------
# 9. Set working directory — project is mounted here at runtime
# -----------------------------------------------------------------------------
WORKDIR /opt/nexusopt

# Use our entrypoint to fix ownership before running the main command
ENTRYPOINT ["/usr/local/bin/nexusopt-entrypoint.sh"]
CMD ["bash", "-c", "sleep infinity"]
