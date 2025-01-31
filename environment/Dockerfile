# hash:sha256:10dfcc2ab38e90d97f26f79f0090aaacb745d5e0aaa818187a4ddc39f8984552
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/mambaforge3:23.1.0-4-python3.10.12-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential=12.9ubuntu3 \
    && rm -rf /var/lib/apt/lists/*

RUN conda install -y \
        mscorefonts \
    && conda clean -ya

RUN pip3 install -U --no-cache-dir \
    code-ocean-aux-tools==1.1.3 \
    numba==0.60.0 \
    pandas[all]==2.2.3 \
    polars==1.17.1 \
    pynwb==2.8.2 \
    s3fs==2024.10.0 \
    scikit-learn==1.5.2 \
    universal-pathlib==0.1.4 \
    zarr==2.17.2 \
    -e git+https://github.com/AllenInstitute/npc_sessions_cache@53a962c924812f3327314931fd3911bf10d8ee15#egg=npc-sessions-cache
