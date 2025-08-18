#!/usr/bin/env bash

# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

set -o errexit -o pipefail -o noclobber -o nounset

if [[ ${DEBUG_BUILD_SCRIPT:-0} -eq 1 ]]; then
    PS4='$LINENO: ' # to see line numbers
    set -x          # to enable debugging
    set -v          # to see actual lines and not just side effects
fi

SCRIPTPATH="$(realpath "$(cat /proc/$$/cmdline | cut --delimiter="" --fields=2)")"
DIR="$(dirname "$SCRIPTPATH")"
PROJECT="$(basename "$DIR")"

DCGM_SKIP_LFS_INSTALL=${DCGM_SKIP_LFS_INSTALL:-0}
DCGM_SKIP_PYTHON_LINTING=${DCGM_SKIP_PYTHON_LINTING:-0}

function die() {
    echo "$@" >&2
    exit 1
}

function usage() {
    echo "Buildscript for the DCGM project
Usage: ${0} [options] [-- [any additional cmake arguments]]
    Where options are:
        -d --debug           : Make debug build of the DCGM (CMAKE_BUILD_TYPE=Debug)
        -r --release         : Make release build of the DCGM (CMAKE_BUILD_TYPE=RelWithDebInfo)
           --coverage        : Generated build dumps coverage information
        -p --packages        : Generate tar.gz packages once the build is done
           --deb             : Generate *.deb packages once the build is done
           --rpm             : Generate *.rpm packages once the build is done
        -c --clean           : Make clean rebuild
        -a --arch <arch>     : Make build for specified architecture. Supported are: amd64, aarch64
        -n --no-tests        : Do not run build-time tests
           --no-install      : Do not perform local installation to the _out directory
           --address-san     : Turn on AddressSanitizer
           --thread-san      : Turn on ThreadSanitizer
           --ub-san          : Turn on UndefinedSanitizer
           --leak-san        : Turn on LeakSanitizer
           --gcc-analyzer    : Turn on GCC static analysis
           --publication     : Generate packages without the BUILD_NUMBER suffix
           --vmware          : Make vmware build of the DCGM
        -h --help            : This information

    Default options are: --release --arch amd64

    Example: ${0} --release --debug --arch amd64 --arch arm64 --clean --packages
        That will build both debug and release binaries for amd64 and arm64 architectures
        and make .tar.gz packages. Output directories will be deleted before builds.
    Example: ${0} --debug -- -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON

    The following ENV VARIABLES may be defined:
    DCGM_DOCKER_IMAGE   : Overrides docker image that will be used for build
    NPROC               : Specifies number of parallel builds allowed. Equals to -jNPROC
    DCGM_SKIP_LFS_INSTALL    : Set to 1 if you want to skip git-lfs verification and installation
    DCGM_SKIP_PYTHON_LINTING : Set to 1 if you do not want running pylint after each build

    Note: Sanitizers (--*-san) will turn off static linking with libstdc++ and non-test packages
          will be disabled as well. Testing packages will contain shared libstdc++ and corresponding
          sanitizer libraries next to the DCGM executables in /usr/share/dcgm_tests/apps/amd64.
          Please also reference https://gcc.gnu.org/onlinedocs/gcc/Instrumentation-Options.html for
          sanitizers compatibility."
}

LONGOPTS=address-san,arch:,clean,coverage,deb,debug,gcc-analyzer,help,leak-san,no-install,no-tests,packages,publication,release,rpm,thread-san,ub-san,vmware
SHORTOPTS=drsa:pchn

! PARSED=$(getopt --options=${SHORTOPTS} --longoptions=${LONGOPTS} --name "${0}" -- "$@")
test ${PIPESTATUS[0]} -eq 0 || die "Failed to parse arguments"

eval set -- "${PARSED}"

declare -a architectures
declare -a build_arguments
declare -a cmake_arguments
declare -a cmake_build_types
declare -a intodocker_arguments

intodocker_arguments=(
    --env DCGM_SKIP_PYTHON_LINTING=$DCGM_SKIP_PYTHON_LINTING)

for VARIABLE in BUILD_NUMBER DEBUG_BUILD_SCRIPT NPROC PRINT_UNCOMMITED_CHANGES
do
    if [[ -n "${!VARIABLE:-}" ]]; then
       intodocker_arguments+=(--env "$VARIABLE=${!VARIABLE}")
    fi
done

CLEAN=0
DEB=0
INSTALL=1
OS=Linux
PUBLICATION=0
RPM=0
TESTS=1
TGZ=0
VMWARE=0

while [[ $# -ne 0 ]]; do
    case "$1" in
        --address-san)
            cmake_arguments+=(-D ADDRESS_SANITIZER=ON)
            ;;
        --arch|-a)
            architectures+=($2)
            shift 2
            continue
            ;;
        --clean|-c)
            CLEAN=1
            ;;
        --coverage)
            cmake_arguments+=(-D DCGM_BUILD_COVERAGE=ON)
            ;;
        --deb)
            DEB=1
            ;;
        --debug|-d)
            cmake_build_types+=(Debug)
            ;;
        --gcc-analyzer)
            cmake_arguments+=(-D GCC_ANALYZER=ON)
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        --leak-san)
            cmake_arguments+=(-D LEAK_SANITIZER=ON)
            ;;
        --no-install)
            INSTALL=0
            ;;
        --no-tests|-n)
            TESTS=0
            ;;
        --packages|-p)
            TGZ=1
            ;;
        --publication)
            PUBLICATION=1
            ;;
        --release|-r)
            cmake_build_types+=(RelWithDebInfo)
            ;;
        --rpm)
            RPM=1
            ;;
        --thread-san)
            cmake_arguments+=(-D THREAD_SANITIZER=ON)
            ;;
        --ub-san)
            cmake_arguments+=(-D UB_SANITIZER=ON)
            ;;
        --vmware)
            cmake_arguments+=(-D VMWARE=ON)
            VMWARE=1
            OS=VMware
            ;;
        --)
            shift
            break
            ;;
        *)
            die "Unrecognized argument: $1"
            ;;
    esac

    build_arguments+=($1)
    shift
done

cmake_arguments+=(-D BUILD_TESTING=$TESTS)

if [[ ${DCGM_BUILD_INSIDE_DOCKER:-0} -eq 0 ]]; then
    if [[ $DCGM_SKIP_LFS_INSTALL -eq 0 ]]; then
        if ! git config --local --get filter.lfs.smudge > /dev/null; then
            echo 'Installing git-lfs locally'
            git-lfs install --local
        fi
        git lfs pull
    fi

    for architecture in "${architectures[@]:-amd64}"; do
        "$DIR/intodocker.sh" "${intodocker_arguments[@]}" \
                             --arch $architecture \
                             -- ./build.sh "${build_arguments[@]}" -- "$@"
    done

    exit
fi

test -v ARCHITECTURE || die "
Required ARCHITECTURE environment variable is not defined in the build
environment. Please check if you are using the proper DCGM build container."

export CMAKE_BUILD_PARALLEL_LEVEL=${NPROC:-$(nproc)}

CMAKE_SOURCE_DIR="$DIR"

declare -A CMAKE_INSTALL_LIBDIR
CMAKE_INSTALL_LIBDIR[DEB]=lib/$TARGET
CMAKE_INSTALL_LIBDIR[RPM]=lib64
CMAKE_INSTALL_LIBDIR[TGZ]=lib/$TARGET

aarch64=aarch64
x86_64=amd64

if command -v ninja > /dev/null; then
    cmake_arguments+=(-G Ninja)
fi

for CMAKE_BUILD_TYPE in "${cmake_build_types[@]:-RelWithDebInfo}"; do
    SUFFIX=$OS-${!ARCHITECTURE}-${CMAKE_BUILD_TYPE,,}
    CMAKE_BINARY_DIR="$DIR/_out/build/$SUFFIX"
    CMAKE_INSTALL_PREFIX="$DIR/_out/$SUFFIX"
    CMAKE_TOOLCHAIN_FILE="$DIR/cmake/$TARGET-toolchain.cmake"

    [[ $CLEAN -eq 0 ]] || rm -rf "$CMAKE_BINARY_DIR" "$CMAKE_INSTALL_PREFIX"
    mkdir --parents "$CMAKE_INSTALL_PREFIX"

    (set -x;
     cmake -S "$CMAKE_SOURCE_DIR" \
           -B "$CMAKE_BINARY_DIR" \
           -U CMAKE_INSTALL_LIBDIR \
           -D CMAKE_BUILD_TYPE=$CMAKE_BUILD_TYPE \
           -D CMAKE_TOOLCHAIN_FILE=$CMAKE_TOOLCHAIN_FILE \
           -D CMAKE_INSTALL_PREFIX=$CMAKE_INSTALL_PREFIX \
           "${cmake_arguments[@]}")

    cp --force "$CMAKE_BINARY_DIR/compile_commands.json" "$CMAKE_SOURCE_DIR/"

    declare -a build_prefix

    (set -x; "${build_prefix[@]}" cmake --build "$CMAKE_BINARY_DIR")

    if [[ $TESTS -eq 1 ]]; then
        if [[ $DCGM_SKIP_PYTHON_LINTING -eq 0 ]]; then
            >&2 echo "Linting Python code"
            pushd "${DIR}/testing/python3"
            export PYLINTHOME="$CMAKE_BINARY_DIR/pylint_out"
            mkdir --parents "$PYLINTHOME"
            find . -type f -name '*.py' | xargs pylint --rcfile pylintrc ../../multi-node/dcgm_multi-node_health_check.py
            popd
        fi

        (set -x; ctest --output-on-failure --test-dir "$CMAKE_BINARY_DIR")
    fi

    test $INSTALL -eq 0 || (set -x; cmake --install "$CMAKE_BINARY_DIR")

    pushd "$CMAKE_BINARY_DIR"

    if [[ $PUBLICATION -eq 1 ]]; then
        unset BUILD_NUMBER
    fi

    for GENERATOR in ${!CMAKE_INSTALL_LIBDIR[@]}; do
        if [[ ${!GENERATOR} -eq 1 ]]; then
            (set -x;
             cmake -S "$CMAKE_SOURCE_DIR" \
                   -B "$CMAKE_BINARY_DIR" \
                   -D CMAKE_INSTALL_LIBDIR=${CMAKE_INSTALL_LIBDIR[$GENERATOR]};
             cmake --build "$CMAKE_BINARY_DIR";
             cpack -G $GENERATOR --verbose)
        fi
    done

    if [[ $VMWARE -eq 0 ]]; then
        pushd dcgm_config

        (set -x;
         cmake -S "$CMAKE_BINARY_DIR/dcgm_config" -B "$CMAKE_BINARY_DIR/dcgm_config";
         cmake --build "$CMAKE_BINARY_DIR/dcgm_config")

        for GENERATOR in RPM DEB; do
            test ${!GENERATOR} -eq 0 || (set -x; cpack -G $GENERATOR)
        done

        popd
    fi

    find . -regex '.*_CPack_Packages.*' -prune \
           -o \( -name '*.tar.gz' -o -name '*.deb' -o -name '*.ddeb' -o -name '*.rpm' \) \
           -exec mv --verbose \
                    --target-directory "$CMAKE_INSTALL_PREFIX" {} +

    find . -name '*rt.props' \
           -exec cp --force \
                    --verbose \
                    --target-directory "$CMAKE_INSTALL_PREFIX" {} +

    popd
done
