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

set -o errexit -o pipefail -o noclobber -o nounset

if [[ ${DEBUG_BUILD_SCRIPT:-} -eq 1 ]]; then
    set -x
fi

if [[ -t 0 ]]; then
    DOCKER_ARGS=-it
fi

ABSPATH=`realpath ${0}`
DIR=$(dirname ${ABSPATH})
PROJECT=$(basename ${DIR})
REMOTE_DIR=/workspaces/${PROJECT}
NPROC=${NPROC:-$(echo $([ -n "$(command -v nproc)" ] && echo $(nproc) || echo 4))}

STATIC_ANALYSIS_MODE_OFF=0
STATIC_ANALYSIS_MODE_ON=1

DCGM_SKIP_PYTHON_LINTING=${DCGM_SKIP_PYTHON_LINTING:-0}

function die() {
    echo "$@" >&2
    exit 1
}

function usage() {
    echo "Buildscript for the DCGM project
Usage: ${0} [options] [-- [any additional cmake arguments]]
    Where options are:
        -d --debug          : Make debug build of the DCGM
        -r --release        : Make release build of the DCGM
           --coverage       : Generated build dumps coverage information
        -p --packages       : Generate tar.gz packages once the build is done
           --deb            : Generate *.deb packages once the build is done
           --rpm            : Generate *.rpm packages once the build is done
        -c --clean          : Make clean rebuild
        -a --arch <arch>    : Make build for specified architecture. Supported are: amd64, ppc64le, aarch64
        -n --no-tests       : Do not run build-time tests
           --no-install     : Do not perform local installation to the _out directory
           --sa-mode <mode> : Run static analysis on build. <mode> := 0|1
           --address-san    : Turn on AddressSanitizer
           --thread-san     : Turn on ThreadSanitizer
           --ub-san         : Turn on UndefinedSanitizer
           --leak-san       : Turn on LeakSanitizer
        -h --help           : This information

    Default options are: --release --arch amd64

    Example: ${0} --release --debug --arch amd64 --arch ppc --clean --packages
        That will build both debug and release binaries for amd64 and ppc64le architectures
        and make .tar.gz packages. Output directories will be deleted before builds.
    Example: ${0} --debug -- -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON

    The following ENV VARIABLES may be defined:
    DCGM_DOCKER_IMAGE   : Overrides docker image that will be used for build
    NPROC               : Specifies number of parallel builds allowed. Equals to -jNPROC
    DCGM_SKIP_PYTHON_LINTING : Set to 1 if you do not want running pylint after each build

    Note: Sanitizers (--*-san) will turn off static linking with libstdc++ and non-test packages
          will be disabled as well. Testing packages will contain shared libstdc++ and corresponding
          sanitizer libraries next to the DCGM executables in /usr/share/dcgm_tests/apps/amd64.
          Please also reference https://gcc.gnu.org/onlinedocs/gcc/Instrumentation-Options.html for
          sanitizers compatibility."
}

function setup_build_env() {
    local arch=${1:-amd64}; shift;
    case ${arch} in
        amd64)
            export TARGET=x86_64-linux-gnu
            ;;
        ppc64le)
            export TARGET=powerpc64le-linux-gnu
            ;;
        aarch64)
            export TARGET=aarch64-linux-gnu
            ;;
        *)
            die "Unsupported architecture"
            ;;
    esac

    export CROSS_PREFIX=/opt/cross
    export TOOLSET_PREFIX=${CROSS_PREFIX}/${TARGET}
    export CC=${CROSS_PREFIX}/bin/${TARGET}-gcc
    export CPP=${CROSS_PREFIX}/bin/${TARGET}-cpp
    export CXX=${CROSS_PREFIX}/bin/${TARGET}-g++
    export LD=${CROSS_PREFIX}/bin/${TARGET}-ld
    export AS=${CROSS_PREFIX}/bin/${TARGET}-as
}

function run_cmake() {
    local output_dir=${1}; shift
    local clean_build=${1:-0}; shift || true
    local packs=${1:-0}; shift || true
    local install_prefix=${1:-}; shift || true
    local runtests=${1:-1}; shift || true
    local static_analysis=${1:-${STATIC_ANALYSIS_MODE_OFF}}; shift || true
    local makedeb=${1:-0}; shift || true
    local makerpm=${1:-0}; shift || true
    local address_san=${1:-0}; shift || true
    local thread_san=${1:-0}; shift || true
    local ub_san=${1:-0}; shift || true
    local leak_san=${1:-0}; shift || true

    if [[ ${static_analysis} -ge ${STATIC_ANALYSIS_MODE_ON} ]] && [[ ! -f /coverity/bin/cov-build ]]; then
        echo "[WARNING] Could not find Coverity. Disabling static analysis" >&2
        static_analysis=${STATIC_ANALYSIS_MODE_OFF}
    fi

    local cmake_install_prefix=""
    if [ -n "${install_prefix}" ]; then
        cmake_install_prefix="-DCMAKE_INSTALL_PREFIX=${install_prefix}"
    fi

    if [[ ${clean_build} -eq 1 ]]; then
        rm -rf ${output_dir} || true
    fi
    mkdir -p ${output_dir}
    pushd ${output_dir}

    local BUILD_TESTING=""
    if [[ ${runtests} -eq 0 ]]; then
        BUILD_TESTING="-DBUILD_TESTING=OFF"
    fi
    local CMAKE_COMMON_ARGS=""
    if [[ ${address_san} -eq 1 ]]; then
        CMAKE_COMMON_ARGS+="-DADDRESS_SANITIZER=ON "
    fi
    if [[ ${thread_san} -eq 1 ]]; then
        CMAKE_COMMON_ARGS+="-DTHREAD_SANITIZER=ON "
    fi
    if [[ ${ub_san} -eq 1 ]]; then
        CMAKE_COMMON_ARGS+="-DUB_SANITIZER=ON "
    fi
    if [[ ${leak_san} -eq 1 ]]; then
        CMAKE_COMMON_ARGS+="-DLEAK_SANITIZER=ON "
    fi
    CMAKE_COMMON_ARGS+="-DCMAKE_TOOLCHAIN_FILE=${DIR}/cmake/${TARGET}-toolchain.cmake ${BUILD_TESTING} ${cmake_install_prefix:-}"
    CMAKE_ARGS="${CMAKE_COMMON_ARGS} ${DIR} $@"

    local cov_emit_dir="$(pwd)/cov_emit_dir"
    local cov_config_args="--config ${cov_emit_dir}/config/coverity-config.xml"
    local build_cmd_prefix=""

    if [[ ${static_analysis} -ge $STATIC_ANALYSIS_MODE_ON ]]; then
        export PATH=/coverity/bin:$PATH
        build_cmd_prefix="cov-build ${cov_config_args} --dir ${cov_emit_dir}"
        mkdir -p ${cov_emit_dir}/config
        cov-configure ${cov_config_args} --template --comptype gcc --compiler x86_64-linux-gnu-gcc
        cov-configure ${cov_config_args} --template --comptype g++ --compiler x86_64-linux-gnu-g++
    fi

    cmake -UDCGM_LIB_INSTALL_PREFIX -UDCGM_PACKAGING -UDCGM_PACKAGING_ENGINE ${CMAKE_ARGS}
    ${build_cmd_prefix} cmake --build . -j${NPROC}

    if [[ ${static_analysis} -ge $STATIC_ANALYSIS_MODE_ON ]]; then
        # Sometimes Coverity leaves its emits in a locked state. Try to analyze them anyway
        rm -f ${cov_emit_dir}/output/*.lock 2>/dev/null || true
        cov-analyze ${cov_config_args} -j auto --dir ${cov_emit_dir} --allow-unmerged-emits \
                    @@${DIR}/dcgm_private/scripts/coverity_analysis_checkers.txt \
            || true # This fails if there are no new compilation units (incremental build). Patch over failure
        rm -rf ${cov_emit_dir}/html 2>/dev/null || true
        cov-format-errors --dir ${cov_emit_dir} --html-output ${cov_emit_dir}/html || true
    fi

    cp compile_commands.json ${DIR}/

    if [[ ${runtests} -eq 1 ]]; then
        if [[ ${DCGM_SKIP_PYTHON_LINTING:-0} == 0 ]]; then
        echo "Linting Python code"
        (cd "${DIR}/testing/python3" && PYLINTHOME="$output_dir/pylint_out" pylint --rcfile pylintrc $(find . -type f -a -iname '*.py'))
        fi
        ctest --output-on-failure
    fi

    if [[ -n "${install_prefix}" ]]; then
        cmake --build . --target install
    fi

    if [[ ${packs} -eq 1 ]]; then
        cpack -G TGZ

        pushd dcgm_config
        cmake ${CMAKE_COMMON_ARGS} . "$@"
        cpack -G TGZ
        mv *.tar.gz ${output_dir}
        popd
    fi

    if [[ ${makedeb} -eq 1 ]]; then
        cmake -DDCGM_PACKAGING=TRUE -DDCGM_PACKAGING_ENGINE=DEB -DDCGM_LIB_INSTALL_PREFIX=lib/${TARGET} ${CMAKE_ARGS}
        cmake --build . -j${NPROC}
        cpack -G DEB

        pushd dcgm_config
        cmake ${CMAKE_COMMON_ARGS} . "$@"
        cmake --build . -j${NPROC}
        cpack -G DEB
        mv *.deb ${output_dir}
        popd
    fi

    if [[ ${makerpm} -eq 1 ]]; then
        cmake -DDCGM_PACKAGING=TRUE -DDCGM_PACKAGING_ENGINE=RPM -DDCGM_LIB_INSTALL_PREFIX=lib64 ${CMAKE_ARGS}
        cmake --build . -j${NPROC}
        cpack -G RPM

        pushd dcgm_config
        cmake ${CMAKE_COMMON_ARGS} . "$@"
        cmake --build . -j${NPROC}
        cpack -G RPM
        mv *.rpm ${output_dir}
        popd
    fi

    popd
}

function install_lfs() {
    local is_lfs_installed=$(git config --local --get filter.lfs.smudge > /dev/null; echo $?)
    if [[ ! ${is_lfs_installed} -eq 0 ]]; then
        echo 'Installing git-lfs locally'
        git-lfs install --local
    fi
    git lfs pull
}

function dcgm_build_within_docker() {
    local debug=${1}; shift
    local release=${1}; shift
    local coverage=${1}; shift
    local arch=${1}; shift
    local packs=${1}; shift
    local clean=${1}; shift
    local runtests=${1}; shift
    local static_analysis=${1}; shift
    local noinstall=${1}; shift
    local makedeb=${1}; shift
    local makerpm=${1}; shift
    local address_san=${1}; shift
    local thread_san=${1}; shift
    local ub_san=${1}; shift
    local leak_san=${1}; shift

    case $( echo "${arch}" | awk '{print tolower($0)}' ) in
        amd64|x64|x86_64)
            arch=amd64
            ;;
        ppc64le|ppc|ppc64)
            arch=ppc64le
            ;;
        aarch64|arm|arm64)
            arch=aarch64
            ;;
        *)
            echo "Unknown architecture: ${arch}"
            exit 1
            ;;
    esac

    setup_build_env ${arch}
    install_lfs

    if [ -x dcgm_private ]; then
        pushd dcgm_private
        install_lfs
        popd
    fi

    local debug_suffix=Linux-${arch}-debug
    local release_suffix=Linux-${arch}-release

    local cwd=$(pwd)
    local outdir_debug=${cwd}/_out/build/${debug_suffix}
    local outdir_release=${cwd}/_out/build/${release_suffix}

    local cmake_args=""

    if [[ ${coverage} -eq 1 ]]; then
        cmake_args=${cmake_args}" -DDCGM_BUILD_COVERAGE=1"
    fi

    local install_prefix=""
    if [[ ${debug} -eq 1 ]]; then
        mkdir -p ${cwd}/_out/${debug_suffix}
        if [[ ${noinstall} -eq 0 ]]; then
            install_prefix=${cwd}/_out/${debug_suffix}
        fi
        run_cmake ${outdir_debug} ${clean} ${packs} "${install_prefix}" ${runtests} ${static_analysis} ${makedeb} ${makerpm} \
            ${address_san} ${thread_san} ${ub_san} ${leak_san} ${cmake_args} -DCMAKE_BUILD_TYPE=Debug "$@"

        find ${outdir_debug} -maxdepth 1 -iname '*.tar.gz' -exec mv {} ${cwd}/_out/${debug_suffix}/ \;
        find ${outdir_debug} -maxdepth 1 -iname '*.deb'    -exec mv {} ${cwd}/_out/${debug_suffix}/ \;
        find ${outdir_debug} -maxdepth 1 -iname '*.rpm'    -exec mv {} ${cwd}/_out/${debug_suffix}/ \;
        find ${outdir_debug} -maxdepth 1 -iname '*rt.props' -exec cp -fv {} ${cwd}/_out/${debug_suffix}/ \;
    fi

    if [[ ${release} -eq 1 ]]; then
        mkdir -p ${cwd}/_out/${release_suffix}
        if [[ ${noinstall} -eq 0 ]]; then
            install_prefix=${cwd}/_out/${release_suffix}
        fi
        run_cmake ${outdir_release} ${clean} ${packs} "${install_prefix}" ${runtests} ${static_analysis} ${makedeb} ${makerpm} \
            ${address_san} ${thread_san} ${ub_san} ${leak_san} ${cmake_args} -DCMAKE_BUILD_TYPE=Release "$@"

        find ${outdir_release} -maxdepth 1 -iname '*.tar.gz' -exec mv {} ${cwd}/_out/${release_suffix}/ \;
        find ${outdir_release} -maxdepth 1 -iname '*.deb'    -exec mv {} ${cwd}/_out/${release_suffix}/ \;
        find ${outdir_release} -maxdepth 1 -iname '*.rpm'    -exec mv {} ${cwd}/_out/${release_suffix}/ \;
        find ${outdir_release} -maxdepth 1 -iname '*rt.props' -exec cp -fv {} ${cwd}/_out/${release_suffix}/ \;
    fi
}

function dcgm_build_using_docker() {
    local debug=${1}; shift
    local release=${1}; shift
    local coverage=${1}; shift
    local arch=${1}; shift
    local packs=${1}; shift
    local clean=${1}; shift
    local runtests=${1}; shift
    local static_analysis=${1}; shift
    local noinstall=${1}; shift
    local makedeb=${1}; shift
    local makerpm=${1}; shift
    local address_san=${1}; shift
    local thread_san=${1}; shift
    local ub_san=${1}; shift
    local leak_san=${1}; shift

    local static_analysis_mount=""
    if [[ ${static_analysis} -ge ${STATIC_ANALYSIS_MODE_ON} ]]; then
        static_analysis_mount="--mount source=dcgm_coverity,target=/coverity"
    fi

    local dcgm_docker_image=$($(dirname $(realpath "${0}"))/intodocker.sh -n)

    local remote_args="--arch ${arch} --sa-mode ${static_analysis}"
    if [[ ${debug} -eq 1 ]]; then
        remote_args="${remote_args} --debug"
    fi
    if [[ ${release} -eq 1 ]]; then
        remote_args="${remote_args} --release"
    fi
    if [[ ${coverage} -eq 1 ]]; then
        remote_args="${remote_args} --coverage"
    fi
    if [[ ${packs} -eq 1 ]]; then
        remote_args="${remote_args} --packages"
    fi
    if [[ ${clean} -eq 1 ]]; then
        remote_args="${remote_args} --clean"
    fi
    if [[ ${runtests} -eq 0 ]]; then
        remote_args="${remote_args} --no-tests"
    fi
    if [[ ${makedeb} -eq 1 ]]; then
        remote_args="${remote_args} --deb"
    fi
    if [[ ${makerpm} -eq 1 ]]; then
        remote_args="${remote_args} --rpm"
    fi
    if [[ ${noinstall} -eq 1 ]]; then
        remote_args="${remote_args} --no-install"
    fi
    if [[ ${address_san} -eq 1 ]]; then
        remote_args="${remote_args} --address-san"
    fi
    if [[ ${thread_san} -eq 1 ]]; then
        remote_args="${remote_args} --thread-san"
    fi
    if [[ ${ub_san} -eq 1 ]]; then
        remote_args="${remote_args} --ub-san"
    fi
    if [[ ${leak_san} -eq 1 ]]; then
        remote_args="${remote_args} --leak-san"
    fi

    docker run --rm -u "$(id -u)":"$(id -g)" \
        ${DOCKER_ARGS:-} \
        -v "${DIR}":"${REMOTE_DIR}" \
        -v /run/docker.sock:/run/docker.sock:rw \
        -w "${REMOTE_DIR}" \
        ${static_analysis_mount} \
        -e "DCGM_BUILD_INSIDE_DOCKER=1" \
        -e "DEBUG_BUILD_SCRIPT=${DEBUG_BUILD_SCRIPT:-0}" \
        -e "NPROC=${NPROC}" \
        -e "BUILD_NUMBER=${BUILD_NUMBER:-}" \
        -e "PRINT_UNCOMMITTED_CHANGES=${PRINT_UNCOMMITTED_CHANGES:-}" \
        -e "DCGM_SKIP_PYTHON_LINTING=${DCGM_SKIP_PYTHON_LINTING:-}" \
        ${dcgm_docker_image} \
        bash -c "set -ex; ./build.sh ${remote_args} -- $@"
}

TARGET_ARCH=()
DEBUG_BUILD=0
RELEASE_BUILD=0
COVERAGE_BUILD=0
PACKAGES_BUILD=0
CLEAN_BUILD=0
RUN_TESTS=1
STATIC_ANALYSIS=${STATIC_ANALYSIS_MODE_OFF}
MAKE_DEB=0
MAKE_RPM=0
NO_INSTALL=0
ADDRESS_SANITIZER=0
THREAD_SANITIZER=0
UB_SANITIZER=0
LEAK_SANITIZER=0

LONGOPTS=debug,release,coverage,arch:,packages,clean,help,no-tests,sa-mode:,deb,rpm,no-install,address-san,thread-san,ub-san,leak-san
SHORTOPTS=dra:pchn

! PARSED=$(getopt --options=${SHORTOPTS} --longoptions=${LONGOPTS} --name "${0}" -- "$@")
if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
    echo "Failed to parse arguments"
    exit 1
fi

eval set -- "${PARSED}"

while true; do
    case "${1}" in
        -d|--debug)
            DEBUG_BUILD=1
            shift
            ;;
        -r|--release)
            RELEASE_BUILD=1
            shift
            ;;
        --coverage)
            COVERAGE_BUILD=1
            shift
            ;;
        -a|--arch)
            TARGET_ARCH+=("${2}")
            shift 2
            ;;
        -p|--packages)
            PACKAGES_BUILD=1
            shift
            ;;
        -c|--clean)
            CLEAN_BUILD=1
            shift
            ;;
        -n|--no-tests)
            RUN_TESTS=0
            shift
            ;;
        --sa-mode)
            case $(echo "${2}" | awk '{print tolower($0)}') in
                ${STATIC_ANALYSIS_MODE_OFF})
                    STATIC_ANALYSIS=${STATIC_ANALYSIS_MODE_OFF}
                    ;;
                ${STATIC_ANALYSIS_MODE_ON})
                    STATIC_ANALYSIS=${STATIC_ANALYSIS_MODE_ON}
                    ;;
                *)
                    echo "Unknown static analysis mode: ${2}"
                    exit 1
                    ;;
            esac
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --deb)
            MAKE_DEB=1
            shift
            ;;
        --rpm)
            MAKE_RPM=1
            shift
            ;;
        --no-install)
            NO_INSTALL=1
            shift
            ;;
        --address-san)
            ADDRESS_SANITIZER=1
            #CLEAN_BUILD=1
            shift
            ;;
        --thread-san)
            THREAD_SANITIZER=1
            #CLEAN_BUILD=1
            shift
            ;;
        --ub-san)
            UB_SANITIZER=1
            CLEAN_BUILD=1
            shift
            ;;
        --leak-san)
            LEAK_SANITIZER=1
            CLEAN_BUILD=1
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

if [[ ${DEBUG_BUILD} -eq 0 ]] && [[ ${RELEASE_BUILD} -eq 0 ]]; then
    RELEASE_BUILD=1
fi

if [[ ${#TARGET_ARCH[@]} -eq 0 ]]; then
    TARGET_ARCH+=(amd64)
fi

for arch in "${TARGET_ARCH[@]}"; do
    if [[ ${DCGM_BUILD_INSIDE_DOCKER:-} -eq 1 ]]; then
        dcgm_build_within_docker ${DEBUG_BUILD} ${RELEASE_BUILD} ${COVERAGE_BUILD} "${arch}" \
            ${PACKAGES_BUILD} ${CLEAN_BUILD} ${RUN_TESTS} ${STATIC_ANALYSIS} ${NO_INSTALL} ${MAKE_DEB} ${MAKE_RPM} \
            ${ADDRESS_SANITIZER} ${THREAD_SANITIZER} ${UB_SANITIZER} ${LEAK_SANITIZER} "$@"
    else
        dcgm_build_using_docker ${DEBUG_BUILD} ${RELEASE_BUILD} ${COVERAGE_BUILD} "${arch}" \
            ${PACKAGES_BUILD} ${CLEAN_BUILD} ${RUN_TESTS} ${STATIC_ANALYSIS} ${NO_INSTALL} ${MAKE_DEB} ${MAKE_RPM} \
            ${ADDRESS_SANITIZER} ${THREAD_SANITIZER} ${UB_SANITIZER} ${LEAK_SANITIZER} "$@"
    fi
done

