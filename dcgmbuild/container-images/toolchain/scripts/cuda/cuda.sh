#!/usr/bin/env bash
#
# Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

set -ex

SYSROOT=/opt/cross/$ARCHITECTURE-linux-gnu/sysroot

for CUDA in cuda11 cuda12 cuda13
do
    read -r _ URL SHA512SUM <<<$(grep "^$CUDA-x86_64-linux-gnu" $1)

    curl --location --fail --output $CUDA-x86_64.deb --retry 5 $URL
    echo "$SHA512SUM $CUDA-x86_64.deb" | sha512sum --check -

    VERSION="$(
        dpkg --field $CUDA-x86_64.deb Version \
        | sed --regexp-extended 's/[^0-9]*([0-9]+[.][0-9]+).*/\1/')"

    VERSION_SUFFIX="${VERSION/./-}"
    HOST_REPO_PACKAGE="$(dpkg --field $CUDA-x86_64.deb Package)"

    dpkg --install $CUDA-x86_64.deb
    cp /var/cuda-repo-*-$VERSION_SUFFIX-local/cuda-*-keyring.gpg /usr/share/keyrings/
    apt update

    rm $CUDA-x86_64.deb

    if [[ $ARCHITECTURE = "x86_64" ]]
    then
        declare -A EXCLUSION_LIST

        while IFS= read -r PACKAGE_NAME
        do
            EXCLUSION_LIST[$PACKAGE_NAME]+=""
        done < <(
            apt-cache depends \
                --recurse \
                --no-breaks \
                --no-conflicts \
                --no-enhances \
                --no-pre-depends \
                --no-recommends \
                --no-replaces \
                --no-suggests \
                build-essential 2> /dev/null \
            | sed '/ .*/d;/<.*/d' \
            | sort)

        while IFS= read -r PACKAGE_NAME
        do
            if [[ -z "${EXCLUSION_LIST[$PACKAGE_NAME]+SKIP}" ]]
            then
                find /var/cuda-repo-*-$VERSION_SUFFIX-local/ \
                     -type f \
                     -name "${PACKAGE_NAME}_*" \
                     -exec dpkg --extract {} / \;
            fi
        done < <(
            apt-cache depends \
                --recurse \
                --no-breaks \
                --no-conflicts \
                --no-enhances \
                --no-pre-depends \
                --no-recommends \
                --no-replaces \
                --no-suggests \
                cuda-cupti-${VERSION_SUFFIX} \
                cuda-nvcc-${VERSION_SUFFIX} \
                cuda-nvml-dev-${VERSION_SUFFIX} \
                libcublas-dev-${VERSION_SUFFIX} \
                libcurand-dev-${VERSION_SUFFIX} 2> /dev/null \
            | sed '/ .*/d;/<.*/d' \
            | sort)

        apt purge --assume-yes $HOST_REPO_PACKAGE

        unset EXCLUSION_LIST

        TARGET=x86_64-linux;
    else
        declare -A HOST_EXCLUSION_LIST
        declare -A TARGET_EXCLUSION_LIST

        while IFS= read -r PACKAGE_NAME
        do
            HOST_EXCLUSION_LIST[$PACKAGE_NAME]+=""
        done < <(
            apt-cache depends \
                --recurse \
                --no-breaks \
                --no-conflicts \
                --no-enhances \
                --no-pre-depends \
                --no-recommends \
                --no-replaces \
                --no-suggests \
                build-essential \
                cuda-cccl-${VERSION_SUFFIX} \
                cuda-crt-${VERSION_SUFFIX} \
                cuda-cudart-${VERSION_SUFFIX} \
                cuda-cudart-dev-${VERSION_SUFFIX} \
                cuda-culibos-dev-${VERSION_SUFFIX} \
                cuda-driver-dev-${VERSION_SUFFIX} 2> /dev/null \
            | sed '/ .*/d;/<.*/d' \
            | sort)

        while IFS= read -r PACKAGE_NAME
        do
            if [[ -z "${HOST_EXCLUSION_LIST[$PACKAGE_NAME]+SKIP}" ]]
            then
                find /var/cuda-repo-*-$VERSION_SUFFIX-local/ \
                     -type f \
                     -name "${PACKAGE_NAME}_*" \
                     -exec dpkg --extract {} / \;

                TARGET_EXCLUSION_LIST[$PACKAGE_NAME]+=""
            fi
        done < <(
            apt-cache depends \
                --recurse \
                --no-breaks \
                --no-conflicts \
                --no-enhances \
                --no-pre-depends \
                --no-recommends \
                --no-replaces \
                --no-suggests \
                cuda-nvcc-${VERSION_SUFFIX} 2> /dev/null \
            | sed '/ .*/d;/<.*/d' \
            | sort)

        apt purge --assume-yes $HOST_REPO_PACKAGE

        read -r _ URL SHA512SUM <<<$(grep "^$CUDA-cross-$ARCHITECTURE-linux-gnu" $1)
        curl --location --fail --output $CUDA-cross.deb --retry 5 $URL
        echo "$SHA512SUM $CUDA-cross.deb" | sha512sum --check -

        HOST_CROSS_REPO_PACKAGE="$(dpkg --field $CUDA-cross.deb Package)"
        CROSS_SUFFIX="$(sed -E 's/cuda-repo-(cross-[^-]+)-.*/\1/' <<< $HOST_CROSS_REPO_PACKAGE)"

        dpkg --install $CUDA-cross.deb
        rm $CUDA-cross.deb

        cp /var/cuda-repo-cross-*-$VERSION_SUFFIX-local/cuda-*-keyring.gpg /usr/share/keyrings/
        apt update

        while IFS= read -r PACKAGE_NAME
        do
            HOST_EXCLUSION_LIST[$PACKAGE_NAME]+=""
        done < <(
            apt-cache depends \
                --recurse \
                --no-breaks \
                --no-conflicts \
                --no-enhances \
                --no-pre-depends \
                --no-recommends \
                --no-replaces \
                --no-suggests \
                cuda-crt-${CROSS_SUFFIX}-${VERSION_SUFFIX} 2> /dev/null \
            | sed '/ .*/d;/<.*/d' \
            | sort)

        while IFS= read -r PACKAGE_NAME
        do
            if [[ -z "${HOST_EXCLUSION_LIST[$PACKAGE_NAME]+SKIP}" ]]
            then
                find /var/cuda-repo-cross-*-$VERSION_SUFFIX-local/ \
                     -type f \
                     -name "${PACKAGE_NAME}_*" \
                     -exec dpkg --extract {} / \;
            fi
        done < <(
            apt-cache depends \
                --recurse \
                --no-breaks \
                --no-conflicts \
                --no-enhances \
                --no-pre-depends \
                --no-recommends \
                --no-replaces \
                --no-suggests \
                cuda-nvcc-${CROSS_SUFFIX}-${VERSION_SUFFIX} 2> /dev/null \
            | sed '/ .*/d;/<.*/d' \
            | sort)

        apt purge --assume-yes $HOST_CROSS_REPO_PACKAGE

        read -r _ URL SHA512SUM <<<$(grep "^$CUDA-$ARCHITECTURE-linux-gnu" $1)
        curl --location --fail --output $CUDA-target.deb --retry 5 $URL
        echo "$SHA512SUM $CUDA-target.deb" | sha512sum --check -

        TARGET_REPO_PACKAGE="$(dpkg --field $CUDA-target.deb Package)"
        DEB_ARCHITECTURE="$(dpkg --field $CUDA-target.deb Architecture)"
        
        dpkg --add-architecture $DEB_ARCHITECTURE
        dpkg --install $CUDA-target.deb
        cp /var/cuda-repo-*-$VERSION_SUFFIX-local/cuda-*-keyring.gpg /usr/share/keyrings/

        CODENAME=$(source /etc/os-release; echo $VERSION_CODENAME)
        cat <<EOF > /etc/apt/sources.list.d/ubuntu.sources
Types: deb
URIs: http://archive.ubuntu.com/ubuntu/
Suites: $CODENAME $CODENAME-updates $CODENAME-backports
Components: main universe restricted multiverse
Architectures: amd64
Signed-By: /usr/share/keyrings/ubuntu-archive-keyring.gpg

Types: deb
URIs: http://security.ubuntu.com/ubuntu/
Suites: $CODENAME-security
Components: main universe restricted multiverse
Architectures: amd64
Signed-By: /usr/share/keyrings/ubuntu-archive-keyring.gpg

Types: deb
URIs: http://ports.ubuntu.com/ubuntu-ports/
Suites: $CODENAME $CODENAME-updates $CODENAME-backports $CODENAME-security
Components: main universe restricted multiverse
Architectures: $DEB_ARCHITECTURE
Signed-By: /usr/share/keyrings/ubuntu-archive-keyring.gpg
EOF

        apt update

        rm $CUDA-target.deb

        while IFS= read -r PACKAGE_NAME
        do
            TARGET_EXCLUSION_LIST[${PACKAGE_NAME%:*}]+=""
        done < <(
            apt-cache depends \
                --recurse \
                --no-breaks \
                --no-conflicts \
                --no-enhances \
                --no-pre-depends \
                --no-recommends \
                --no-replaces \
                --no-suggests \
                build-essential:${DEB_ARCHITECTURE} 2> /dev/null \
            | sed '/ .*/d;/<.*/d' \
            | sort)

        while IFS= read -r PACKAGE_NAME
        do
            PACKAGE_NAME=${PACKAGE_NAME%:*}
            if [[ -z "${TARGET_EXCLUSION_LIST[$PACKAGE_NAME]+SKIP}" ]]
            then
                echo $PACKAGE_NAME
                find /var/cuda-repo-*-$VERSION_SUFFIX-local/ \
                     -type f \
                     -name "${PACKAGE_NAME}_*" \
                     -exec dpkg --extract {} / \;
            fi
        done < <(
            apt-cache depends \
                --recurse \
                --no-breaks \
                --no-conflicts \
                --no-enhances \
                --no-pre-depends \
                --no-recommends \
                --no-replaces \
                --no-suggests \
                cuda-cupti-${VERSION_SUFFIX}:${DEB_ARCHITECTURE} \
                cuda-nvcc-${VERSION_SUFFIX}:${DEB_ARCHITECTURE} \
                cuda-nvml-dev-${VERSION_SUFFIX}:${DEB_ARCHITECTURE} \
                libcublas-dev-${VERSION_SUFFIX}:${DEB_ARCHITECTURE} \
                libcurand-dev-${VERSION_SUFFIX}:${DEB_ARCHITECTURE} 2> /dev/null \
            | sed '/ .*/d;/<.*/d' \
            | sort)

        apt purge --assume-yes $TARGET_REPO_PACKAGE
        dpkg --remove-architecture $DEB_ARCHITECTURE

        TARGET=${CROSS_SUFFIX#*-}-linux;

        unset HOST_EXCLUSION_LIST
        unset TARGET_EXCLUSION_LIST
    fi

    while IFS= read -r ENTRY
    do
        if [[ -f $ENTRY ]]
        then
             ln $ENTRY $SYSROOT/$ENTRY
        elif [[ -d $ENTRY ]];
        then
            mkdir --parents $SYSROOT/$ENTRY
        elif [[ -L $ENTRY ]];
        then
            case $(readlink $ENTRY) in
              /*)
                  ln --symbolic $SYSROOT/$(realpath $ENTRY) $SYSROOT/$ENTRY
                  ;;
              *)
                  pushd $SYSROOT/$(dirname $ENTRY)
                  ln --symbolic $(readlink $ENTRY) $(basename $ENTRY)
                  popd
                  ;;
            esac
        fi
    done < <(find /usr/local/cuda-$VERSION/targets/$TARGET | sort)
    
    pushd $SYSROOT/usr/local/cuda-$VERSION/
    ln --symbolic targets/$TARGET/include include
    ln --symbolic targets/$TARGET/lib lib64
    popd

    VERSION_MAJOR=${VERSION%.*}
    for ENTRY in "" $SYSROOT
    do
        pushd $ENTRY/usr/local 
        ln --symbolic cuda-$VERSION cuda-$VERSION_MAJOR
        popd
    done

    cat <<EOF > $SYSROOT/../toolchain-$CUDA.cmake
#
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
#
include("\${CMAKE_CURRENT_LIST_DIR}/toolchain.cmake")

set(CMAKE_CUDA_COMPILER "/usr/local/cuda-$VERSION_MAJOR/bin/nvcc")
set(CMAKE_CUDA_COMPILER_TARGET "\${CMAKE_LIBRARY_ARCHITECTURE}")
set(CMAKE_CUDA_HOST_COMPILER "\${CMAKE_CXX_COMPILER}")

string(JOIN " " CMAKE_CUDA_FLAGS_INIT
    "-ccbin=\${CMAKE_CXX_COMPILER}"
    "--target-directory $TARGET"
    "-Xcompiler=-static-libgcc"
    "-Xcompiler=-static-libstdc++")
EOF

    cp $SYSROOT/../toolchain.meson.txt $SYSROOT/../toolchain-$CUDA.meson.txt
    cat <<EOF >> $SYSROOT/../toolchain-$CUDA.meson.txt

[binaries]
cuda = [
    '/usr/local/cuda-$VERSION_MAJOR/bin/nvcc',
    '-ccbin=' + crossroot / 'bin' / target + '-g++',
    '--target-directory=$TARGET']

[built-in options]
cuda_ld = crossroot / 'bin' / target + '-ld' 
cuda_link_args = [
    '-Xcompiler=-static-libgcc',
    '-Xcompiler=-static-libstdc++'] + linker_paths
EOF
done

apt autoremove --purge --quiet --assume-yes
apt clean --quiet --assume-yes
rm -rf /var/lib/apt/lists/*
