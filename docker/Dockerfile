# syntax=docker/dockerfile:1.7
# Per-tool micromamba image for the 4-week-cut GFP pipeline.
# One CUDA runtime base; one isolated micromamba env per tool. Each env is built
# from that tool's own manifest where one exists, so pins are not hand-merged.
#
# Target: NVIDIA V100S 32 GB (sm_70), deployed via nvidia-enroot/pyxis.
# Per-env torch wheels carry their own CUDA runtime; the base only supplies the
# driver (passed through by enroot). That lets spurs (torch 1.12/cu113) coexist
# with everyone else (torch 2.4/cu121) on the same image, in separate envs.

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    MAMBA_ROOT_PREFIX=/opt/micromamba \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      git curl ca-certificates bzip2 build-essential procps \
    && rm -rf /var/lib/apt/lists/*

# micromamba
RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
      | tar -xvj -C /usr/local bin/micromamba

# ---- vendored repos (read-only). .dockerignore must strip weights/data/*.pt ----
WORKDIR /repos
COPY repos/ProteinMPNN   /repos/ProteinMPNN
COPY repos/LigandMPNN    /repos/LigandMPNN
COPY repos/ThermoMPNN-D  /repos/ThermoMPNN-D
COPY repos/SPURS         /repos/SPURS
COPY repos/evodiff       /repos/evodiff
COPY repos/esm           /repos/esm
COPY envs/mamba          /repos/envs

# ---- env: proteinmpnn (ProteinMPNN + ProteinMPNN-ddG) ----
RUN micromamba create -y -n proteinmpnn -f /repos/envs/proteinmpnn.yml && \
    micromamba clean -ay

# ---- env: ligandmpnn (built from the repo's own requirements) ----
RUN micromamba create -y -n ligandmpnn python=3.10 pip && \
    micromamba run -n ligandmpnn pip install \
      --extra-index-url https://download.pytorch.org/whl/cu121 torch==2.4.1 && \
    micromamba run -n ligandmpnn pip install -r /repos/LigandMPNN/requirements.txt && \
    micromamba clean -ay

# ---- env: thermompnn (built from the repo's own conda env file) ----
RUN micromamba create -y -n thermompnn -f /repos/ThermoMPNN-D/environment.yaml && \
    micromamba clean -ay

# ---- env: spurs (old torch 1.12 / PL 1.7 / cu113 — fully isolated) ----
# SPURS pins 2022-era wheels (pytorch-lightning 1.7.3, torchdata 0.4.0, ...) whose
# metadata violates PEP 440 ("torch (>=1.9.*)"). pip>=24.1 validates metadata
# strictly and rejects them, so this env must use pip<24.1.
RUN micromamba create -y -n spurs python=3.10 "pip<24.1" && \
    micromamba run -n spurs pip install \
      --extra-index-url https://download.pytorch.org/whl/cu113 torch==1.12.0 && \
    micromamba run -n spurs pip install -r /repos/SPURS/requirements.txt && \
    micromamba run -n spurs pip install -e /repos/SPURS && \
    micromamba clean -ay

# ---- env: esm (Biohub ESM SDK v3.3.0 — ESMC 6B + ESMFold2) ----
# env python=3.12 already satisfies esm's requires-python (>=3.12,<3.13), so no
# pin patch is needed. `pip install -e` also clones the Biohub transformers fork
# (provides transformers.models.esmfold2), so git + network are required here.
RUN micromamba create -y -n esm -f /repos/envs/esm.yml && \
    micromamba run -n esm pip install -e /repos/esm && \
    micromamba clean -ay

# ---- env: evodiff (fair-esm based — isolated from the esm SDK name clash) ----
RUN micromamba create -y -n evodiff python=3.10 pip && \
    micromamba run -n evodiff pip install \
      --extra-index-url https://download.pytorch.org/whl/cu121 torch==2.4.1 && \
    micromamba run -n evodiff pip install -e /repos/evodiff && \
    micromamba clean -ay

# ---- env: dnatools (CPU; bioconda viennarna + foldseek) ----
RUN micromamba create -y -n dnatools -f /repos/envs/dnatools.yml && \
    micromamba clean -ay

# ---- post-build env patches ----
# Kept as one trailing layer on purpose: a rebuild reuses every cached env layer
# above and only re-runs this cheap step. Three runtime breakages:
#  - thermompnn: ThermoMPNN-D/environment.yaml has a self-inconsistent CUDA stack
#    (pytorch-cuda=11.7 alongside cudatoolkit=11.3) and floats mkl to 2025, so the
#    conda torch fails twice: libtorch_cpu.so wants the ITT symbol iJIT_NotifyEvent
#    that mkl>=2024.1 dropped (pytorch#123097), and libtorch_cuda.so wants
#    cudaGraphInstantiateWithFlags (CUDA 11.4+), absent from cudatoolkit 11.3.
#    Replace the conda torch with the self-contained cu121 pip wheel (the same one
#    ligandmpnn/evodiff/esm use on the V100S); it bundles its own MKL + cudart, so
#    both conda-side mismatches stop mattering.
#  - spurs:   pytorch-lightning 1.7.3 imports pkg_resources, removed in setuptools>=81.
#  - evodiff: evodiff/pretrained.py imports pkg_resources, same setuptools>=81 cause.
RUN micromamba run -n thermompnn pip install \
      --extra-index-url https://download.pytorch.org/whl/cu121 torch==2.4.1 && \
    micromamba run -n spurs   pip install "setuptools<81" && \
    micromamba run -n evodiff pip install "setuptools<81" && \
    micromamba clean -ay

# ---- launchers ----
COPY synenv.sh    /usr/local/bin/synenv.sh
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
COPY synrun       /usr/local/bin/synrun
RUN chmod +x /usr/local/bin/entrypoint.sh /usr/local/bin/synrun

ENV WORKDIR=/workspace
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["bash"]
