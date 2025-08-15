#!/usr/bin/env bash

# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

set -o errexit -o pipefail -o nounset

if [[ ${DEBUG_BUILD_SCRIPT:-} -eq 1 ]]; then
    PS4='$LINENO: ' # to see line numbers
    set -x          # to enable debugging
    set -v          # to see actual lines and not just side effects
fi

###################
#### Arguments ####
###################

ARCHITECTURES=("x86_64" "aarch64")
if [[ $# -gt 0 ]]; then
    ARCHITECTURES=("$@")
fi

####################
#### Versioning ####
####################

if [[ ! -v TAG ]]
then
  TAG=${DCGM_VERSION:-4.0.0}-gcc14-$(scripts/toolchain-sha256sum | head -c6)
fi

###############
#### Logic ####
###############

export BASE_IMAGE=${BASE_IMAGE:-ubuntu:24.04}
export TAG

docker compose build dcgm-common-host-software
for ARCHITECTURE in "${ARCHITECTURES[@]}"
do
  docker compose build dcgm-toolchain-$ARCHITECTURE
  docker compose build dcgmbuild-$ARCHITECTURE
  docker tag ${REGISTRY:-dcgm}/dcgmbuild:$TAG-$ARCHITECTURE dcgmbuild:$TAG-$ARCHITECTURE
done

docker image rm dcgm/common-host-software:$TAG
