# Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
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

FROM nvcr.io/nvidia/cuda:13.3.0-base-ubi10

LABEL io.k8s.display-name="NVIDIA DCGM"
LABEL name="NVIDIA DCGM"
LABEL vendor="NVIDIA"
LABEL release="N/A"
LABEL summary="Manage NVIDIA GPUs"
LABEL description="See summary"

ARG DCGM_VERSION
ARG PACKAGE_REVISION
LABEL version="${DCGM_VERSION}-${PACKAGE_REVISION}"

RUN dnf update -y && \
    dnf upgrade -y && \
    dnf install -y \
        libcap \
        datacenter-gpu-manager-4-cuda12-${DCGM_VERSION}-${PACKAGE_REVISION} \
        datacenter-gpu-manager-4-cuda13-${DCGM_VERSION}-${PACKAGE_REVISION} \
        && \
    dnf autoremove -y \
        openssl \
        && \
    dnf clean all

# Required for full gpu access
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,compat32
# Disable all constraints on the configurations required by NVIDIA container toolkit
ENV NVIDIA_DISABLE_REQUIRE="true"
ENV NVIDIA_VISIBLE_DEVICES=all

ENV NO_SETCAP=
# open port 5555 to other containers ie, dcgm-exporter
EXPOSE 5555

ENTRYPOINT ["/usr/bin/nv-hostengine"]
CMD ["-n", "-b", "0.0.0.0", "--log-level", "NONE", "-f", "-"]
