#!/usr/bin/env bash

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

set -e -o pipefail -o nounset

CUR_DIR=$(dirname "$(realpath "${0}")")
PRE_COMMIT_FILE=${CUR_DIR}/.git/hooks/pre-commit
readonly HOOKS=(
    "./validate_format.sh"
    "./validate_clippy.sh"
)

if [ ! -f "${PRE_COMMIT_FILE}" ]; then
    echo "#!/usr/bin/env bash" > "${PRE_COMMIT_FILE}"
    echo "set -e" >> "${PRE_COMMIT_FILE}"
    echo "" >> "${PRE_COMMIT_FILE}"
    chmod +x "${PRE_COMMIT_FILE}"
fi

installed=0
for hook in "${HOOKS[@]}"; do
    if grep -Fx "${hook}" "${PRE_COMMIT_FILE}" >/dev/null; then
        continue
    fi

    echo "${hook}" >> "${PRE_COMMIT_FILE}"
    installed=1
done

if [[ ${installed} -eq 0 ]]; then
    echo "Pre-commit hooks are already installed"
    exit 0
fi
