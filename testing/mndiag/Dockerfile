# syntax=docker/dockerfile:1.4
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

ARG BASE_IMAGE=ubuntu:24.04
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive

# OpenMPI for mpirun, openssh-server for inter-container SSH
# Cache mounts reuse downloaded apt packages across builds
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && \
    apt-get install -y \
        openmpi-bin \
        openssh-server \
        python3 \
        python3-pip \
        python3-setuptools \
        file \
        procps \
        iproute2

# To satisfy framework warnings
# Cache mount reuses pip wheels across builds
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install mock matplotlib --break-system-packages

# Stub nvidia-smi so test containers don't emit "command not found" or XML parse noise
COPY --chmod=755 <<'EOF' /usr/local/bin/nvidia-smi
#!/bin/sh
case "$*" in
    *-q*-x*|*-x*-q*) echo '<?xml version="1.0" ?><nvidia_smi_log></nvidia_smi_log>' ;;
esac
exit 0
EOF

# Setup required for SSH server
RUN mkdir /var/run/sshd && \
    ssh-keygen -A && \
    mkdir -p -m 700 /root/.ssh && \
    echo "/opt/dcgm/lib/" > /etc/ld.so.conf.d/dcgm.conf

# Configure SSH server to allow passwordless root login
COPY <<EOF /etc/ssh/sshd_config.d/mndiag.conf
PermitRootLogin yes
PasswordAuthentication no
PubkeyAuthentication yes
AuthorizedKeysFile .ssh/authorized_keys
EOF

# Configure SSH client to disable host key prompt
COPY --chmod=600 <<EOF /root/.ssh/config
Host *
    StrictHostKeyChecking no
EOF

# Shared key pair baked into the image, all containers trust each other automatically
RUN ssh-keygen -t ed25519 -N "" -f /root/.ssh/id_ed25519 && \
    cat /root/.ssh/id_ed25519.pub >> /root/.ssh/authorized_keys && \
    chmod 600 /root/.ssh/authorized_keys

WORKDIR /workspace

# Keep the container running and listening for MPI/SSH connections
# Run ldconfig at startup so dynamic linker caches volume mounted libraries
CMD ldconfig && /usr/sbin/sshd -D
