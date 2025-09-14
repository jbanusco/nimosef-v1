# CUDA 12.1 + cuDNN 8 + Ubuntu 20.04
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu20.04

LABEL maintainer="Jaume Banus <jaume.banus-cobo@chuv.ch>"

ENV DEBIAN_FRONTEND=noninteractive

# -------------------------------------------------------------------------
# Install Python 3.10 + system deps
# -------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    wget git curl ca-certificates g++ procps vim \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y python3.10 python3.10-distutils python3.10-dev \    
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Install pip
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3

# Upgrade pip & setuptools
RUN pip install --upgrade pip setuptools wheel

# -------------------------------------------------------------------------
# Set CUDA env vars
# -------------------------------------------------------------------------
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
ENV DGLBACKEND=pytorch

# -------------------------------------------------------------------------
# Copy project
# -------------------------------------------------------------------------
WORKDIR /usr/src/
COPY . /usr/src/

# -------------------------------------------------------------------------
# Install GPU stack + project
# -------------------------------------------------------------------------
RUN chmod +x install_gpu.sh && ./install_gpu.sh

# -------------------------------------------------------------------------
# Entrypoint
# -------------------------------------------------------------------------
CMD ["python3"]
