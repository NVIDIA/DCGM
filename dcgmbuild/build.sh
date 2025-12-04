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

GIT_COMMIT=$(git rev-parse HEAD 2>/dev/null)
if [[ $? -eq 0 ]]
then
  if ! git diff-index --exit-code --quiet HEAD
  then
    GIT_COMMIT+='-dirty'
  fi
fi

###############
#### Logic ####
###############

export BASE_IMAGE=${BASE_IMAGE:-ubuntu:24.04}
export GIT_COMMIT
export TAG=${TAG:=latest}

docker compose build dcgm-common-host-software
for ARCHITECTURE in "${ARCHITECTURES[@]}"
do
  docker compose build dcgm-toolchain-$ARCHITECTURE
  docker compose build dcgmbuild-$ARCHITECTURE
done

docker image rm dcgm/common-host-software:$TAG
