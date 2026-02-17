#!/usr/bin/env bash

# Copyright (c) 2023-2026, NVIDIA CORPORATION.  All rights reserved.
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

set -e -o pipefail -o nounset -x

if [[ ${DCGM_BUILD_INSIDE_DOCKER:-0} -eq 0 ]]; then
    "$(dirname "$(realpath "${0}")")/intodocker.sh" --pull -- "${0}" "$@"
    exit $?
fi

files_from=staged

readonly BATCH_SIZE=$(nproc)

# Python exclusion pattern matching .pep8 config and autopep8 defaults
# Manual exclusions required: autopep8 has no --assume-filename equivalent for stdin
readonly PYTHON_EXCLUDE_PATTERN='(^|/)(_out|sdk|PerfWorks|__pycache__)(/|$)|testing/python3/libs_3rdparty/|(^|/)\.'

function GetChangedFiles(){
    case ${files_from} in
        staged)
            git diff --cached --name-only --diff-filter=d
            ;;
        merged)
            git diff-tree --name-only --diff-filter=d -r -m --no-commit-id HEAD
            ;;
        *)
            return
            ;;
    esac
}

function GetCppFiles(){
    GetChangedFiles | grep -E "\.(cpp|h|hpp|c)$" || true
}

function DetectPythonFiles(){
    while IFS= read -r file
    do
        if (git show :"${file}" 2>/dev/null | file - | grep -q 'Python script')
        then
            printf '%s\n' "${file}"
        fi
    done
}

function GetPythonFiles(){
    local all_files=$(GetChangedFiles | grep -vE "${PYTHON_EXCLUDE_PATTERN}" || true)

    echo "${all_files}" | grep -E "\.py$" || true
    echo "${all_files}" | grep -vE "\.py$" | DetectPythonFiles || true
}

function GetFileContent(){
    local filename="${1}"
    git show :"${filename}"
}

# Generic file validation function
# Args: $1=language_name, $2=format_command; reads file list from stdin
function ValidateFiles(){
    local language_name="${1}"
    local format_command="${2}"
    local -a files=()

    while IFS= read -r file; do
        pushd "$(dirname "$(realpath "${file}")")" >/dev/null
        if ! cmp -s <(GetFileContent "${file}") <(GetFileContent "${file}" | eval "${format_command}"); then
            files+=("${file}")
        fi
        popd >/dev/null
    done

    if [[ ${#files[@]} -gt 0 ]]; then
        echo "${language_name} format error within the following files:" >&2
        printf "%s\n" "${files[@]}" >&2

        if [ ${show_diff} -eq 1 ]; then
            for file in "${files[@]}"; do
                pushd "$(dirname "$(realpath "${file}")")" >/dev/null
                diff -au --label "${file}" <(GetFileContent "${file}") --label "${file} (Formatted)" \
                    <(GetFileContent "${file}" | eval "${format_command}") || true
                popd >/dev/null
                echo ""
            done
        fi

        return 1
    fi

    return 0
}

function ValidateCppFiles(){
    GetCppFiles | ValidateFiles "C/C++" "clang-format --style=file --assume-filename=\${file}"
}

function ValidatePythonFiles(){
    if ! command -v autopep8 &> /dev/null; then
        echo "WARNING: autopep8 not found. Skipping Python file validation." >&2
        return 0
    fi

    GetPythonFiles | ValidateFiles "Python" "autopep8 -"
}

function usage() {
    echo "Formatting validation script for DCGM project
Usage: ${0} [options]
    Where options are:
        -d --diff       : Show diff for files which does not complay
        -m --merged     : Check merged files instead of staged files (for Jenkins)
        -h --help       : This information

    Without --merged argument this script will validate staged C/CPP/H/HPP and Python files."
}

LONG_OPTS=merged,diff,help
SHORT_OPTS=m,d,h

if ! PARSED=$(getopt --options=${SHORT_OPTS} --longoptions=${LONG_OPTS} --name "${0}" -- "$@"); then
    echo "Failed to parse arguments" >&2
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
            echo "Wrong arguments" >&2
            exit 1
            ;;
    esac
done

has_errors=0

if ! ValidateCppFiles; then
    has_errors=1
fi

if ! ValidatePythonFiles; then
    has_errors=1
fi

exit $has_errors
