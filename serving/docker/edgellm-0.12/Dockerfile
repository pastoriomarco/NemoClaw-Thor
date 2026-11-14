# TensorRT Edge-LLM 0.12.0 runtime for Jetson Thor (JetPack 7.2.1: CUDA 13.2, TensorRT 10.16.2.10)
FROM nvidia/cuda:13.2.1-runtime-ubuntu24.04
ARG TRT_VER=10.16.2.10-1+cuda13.2
# JetPack TensorRT comes from the Jetson repo (the platform-provided build), not the CUDA sbsa repo
COPY jetson-ota-public.asc /etc/apt/trusted.gpg.d/jetson-ota-public.asc
RUN echo "deb https://repo.download.nvidia.com/jetson/common r39.2 main" > /etc/apt/sources.list.d/nvidia-l4t-apt-source.list
RUN set -eux; apt-get update; \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      python3 python3-venv python3-pip ca-certificates \
      libnvinfer10=${TRT_VER} libnvinfer-plugin10=${TRT_VER} libnvinfer-lean10=${TRT_VER} \
      libnvinfer-dispatch10=${TRT_VER} libnvinfer-vc-plugin10=${TRT_VER} libnvonnxparsers10=${TRT_VER} \
      python3-libnvinfer=${TRT_VER}; \
    rm -rf /var/lib/apt/lists/*
RUN python3 -m venv --system-site-packages /opt/edgellm && \
    /opt/edgellm/bin/pip install --no-cache-dir "tensorrt-edgellm[runtime]==0.12.0"
ENV PATH=/opt/edgellm/bin:$PATH
RUN python -c "import tensorrt; print(tensorrt.__version__)"
WORKDIR /workspace
