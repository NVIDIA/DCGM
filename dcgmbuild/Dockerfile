#
# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
#
ARG BASE_IMAGE
FROM $BASE_IMAGE as stage1

ARG DEBIAN_FRONTEND=noninteractive

COPY --chown=root:root scripts /root/.build/scripts/
RUN set -ex; find /root/.build/scripts -iname '*.sh' -exec chmod a+x {} \;
WORKDIR /root/.build/scripts

ARG BASE_IMAGE_TARGET
RUN bash -c 'set -ex -o pipefail; export TARGET=$1-linux-gnu; find . -iregex "^\.\/[0-9]+_.*" | sort | xargs -n1 -I {} bash -c "{} || exit 255"' bash $BASE_IMAGE_TARGET

FROM $BASE_IMAGE as final
COPY --from=stage1 /opt /opt

RUN mkdir -p /workspaces
RUN chmod a+rw /workspaces