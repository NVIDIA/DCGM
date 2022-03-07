#!/usr/bin/env bash

# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

if [[ ${DCGM_BUILD_INSIDE_DOCKER:-0} -eq 0 ]]; then
    $(dirname $(realpath "${0}"))/intodocker.sh --pull -- "${0}" "$@"
    exit $?
fi

files_from=staged

function GetCppFiles(){
    local git_top_dir=$(pwd)
    case ${files_from} in
        staged)
            local top_files=$(git diff --cached --name-only --diff-filter=d | grep -E "\.(cpp|h|hpp|c)$")
            local sub_files=$(git submodule foreach --quiet --recursive "git diff --cached --name-only --diff-filter=d | grep -E \"\.(cpp|h|hpp|c)$\" | xargs -r -n1 realpath --relative-to=${git_top_dir}")
            echo "${top_files}"$'\n'"${sub_files}"
            ;;
        merged)
            local top_files=$(git diff-tree --name-only --diff-filter=d -r -m --no-commit-id HEAD | grep -E "\.(cpp|h|hpp|c)$")
            local sub_files=$(git submodule foreach --quiet --recursive "git diff-tree --name-only --diff-filter=d -r -m --no-commit-id HEAD | grep -E \"\.(cpp|h|hpp|c)$\" | xargs -r -n1 realpath --relative-to=${git_top_dir}")
            echo "${top_files}"$'\n'"${sub_files}"
            ;;
        *)
            ;;
    esac
}

function GetFileContent(){
    local filename=$(basename "${1}")
    git show :"./${filename}"
}

function usage() {
    echo "Formatting validation script for DCGM project
Usage: ${0} [options]
    Where options are:
        -d --diff       : Show diff for files which does not complay
        -m --merged     : Check merged files instead of staged files (for Jenkins)
        -h --help       : This information

    Without --merged argument this script will validate changed AND staged C/CPP/H/HPP files."
}

LONG_OPTS=merged,diff,help
SHORT_OPTS=m,d,h

! PARSED=$(getopt --options=${SHORT_OPTS} --longoptions=${LONG_OPTS} --name "${0}" -- "$@")
if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
    echo "Failed to parse arguments"
    exit 1
fi

eval set -- "${PARSED}"

show_diff=0
while true; do
    case "${1}" in
        -h|--help)
            usage
            exit 0
            ;;
        -m|--merged)
            files_from=merged
            shift
            ;;
        -d|--diff)
            show_diff=1
            shift
            ;;
        --)
            shift
            break
            ;;
        *)
            echo "Wrong arguments"
            exit 1
            ;;
    esac
done

input_files=$(GetCppFiles || true)

files=()
for file in ${input_files}; do
    pushd $(dirname $(realpath ${file})) >/dev/null
    if ! cmp -s <(GetFileContent ${file}) <(GetFileContent ${file} | clang-format --style=file --assume-filename=${file}); then
        files+=("${file}")
    fi
    popd >/dev/null
done

if [[ ${#files[@]} -gt 0 ]]; then
    echo Format error within the following files:
    printf "%s\n" "${files[@]}"

    if [ ${show_diff} -eq 1 ]; then
        for file in ${files[@]}; do
            pushd $(dirname $(realpath ${file})) >/dev/null
            diff -au --label ${file} <(GetFileContent ${file}) --label "${file} (Formatted)" <(GetFileContent ${file} | \
                clang-format --style=file --assume-filename=${file}) || true
            popd >/dev/null
            echo ""
        done
    fi

    exit 1
fi
