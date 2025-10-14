ARG BASE_IMAGE=ubuntu:20.04
FROM $BASE_IMAGE

ARG CMAKE_VERSION=3.24.4

ENV NVIDIA_VISIBLE_DEVICES="all" \
    NVIDIA_DRIVER_CAPABILITIES="compute,utility" 

RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        gnupg2 \
        curl \
        ca-certificates \
        libxml2 \
        build-essential \
        libboost-program-options-dev \
        numactl && \
    apt clean && \
    rm --recursive --force /var/lib/apt/lists/*

# ubuntu 20.04 has default cmake 3.16, nvbandwidth requires cmake >= 3.20
# Download and install CMake
RUN curl --location \
         --output cmake.sh \
         https://github.com/Kitware/CMake/releases/download/v$CMAKE_VERSION/cmake-$CMAKE_VERSION-linux-$(uname -p).sh && \
    chmod +x cmake.sh && \
    ./cmake.sh --skip-license --prefix=/usr/local && \
    rm cmake.sh

RUN --mount=type=bind,source=cuda-repo.deb,target=/mnt/host/cuda-repo.deb \
    export DEBIAN_FRONTEND=noninteractive && \
    PACKAGE=$(dpkg-deb --field /mnt/host/cuda-repo.deb Package) && \
    dpkg --install /mnt/host/cuda-repo.deb && \
    cp /var/$PACKAGE/cuda-*-keyring.gpg /usr/share/keyrings/ && \
    apt-get update && \
    apt-get install --yes --no-install-recommends \
        $(apt-cache search cuda-cudart | awk '{print $1}') \
        $(apt-cache search cuda-nvcc | awk '{print $1}') \
        $(apt-cache search cuda-nvml | awk '{print $1}') && \
    apt remove --yes $PACKAGE && \
    rm --recursive --force /var/lib/apt/lists/*
    
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/cuda/lib64/stubs:${LD_LIBRARY_PATH}"

