#!/usr/bin/env bash
# Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

VERSION=21
read -r _ URL SHA512SUM <<<$(grep '^clang ' $1)

export DEBIAN_FRONTEND=noninteractive
apt install --quiet --assume-yes software-properties-common gpg lsb-release wget

curl --location --fail --output llvm.sh --retry 5 $URL
echo "$SHA512SUM llvm.sh" | sha512sum --check -

chmod +x llvm.sh
./llvm.sh $VERSION all

apt autoremove
apt clean
rm -rf /var/lib/apt/lists/*

# Establish an alternatives group for binaries provided by LLVM packages.
#
# LLVM packages install files to the `/usr/lib/llvm-<MAJOR VERSION>` directory
# and configure symbolic links to executable binaries therein in the `/usr/bin`
# directory. For better or worse, these symbolic links include the LLVM major
# version as a suffix.
#
# This accommodates having multiple major versions of LLVM tools available in
# the executable path at a given time, which can be useful for persistent shared
# environments supporting multiple teams. However, for environments providing a
# single such suite of tools (such as a build container), it necessitates
# coordinating updates of the build image and maintainers/developers of
# downstream products using that image.
#
# To avoid this coordination for the DCGM build image, we configure supplemental
# symbolic links _without_ the major version suffix. This implemented and
# managed using the update-alternatives system.
#
# Here we configure an alternative group for managing the precedence of these
# LLVM-related symbolic links, using `clang` as the main alternative and
# registering all other alternatives as subordinate/slave links. In this way, a
# single priority describes the entire group and can be updated atomically.
#
# See the update-alternatives manual page for more information.
# https://man7.org/linux/man-pages/man1/update-alternatives.1.html
#
ALTERNATIVES=(
     "/usr/bin/clang clang $(readlink -f /usr/bin/clang-$VERSION) $VERSION")

# For each symbolic link referring to an executable file in the LLVM
# installation directory (aside for clang itself), append an associated
# subordinate alternative link to the ALTERNATIVES array

readarray -t -O 1 ALTERNATIVES < <(
    find /usr/bin -type l \
                  -xtype f \
                  -name "*-$VERSION" \
                  -not -name "clang-$VERSION" \
                  -lname "../lib/llvm-$VERSION/bin/*" \
                  -regextype posix-extended \
                  -regex '[^ ]*' \
    | while IFS= read -r FILEPATH; do
        TARGET=$(readlink -f $FILEPATH)

        [[ -x $TARGET ]] || continue

        ALTERNATIVE_PATH=${FILEPATH%-*}
        ALTERNATIVE_NAME=$(basename $ALTERNATIVE_PATH)

        printf -- '--slave %s %s %s\n' $ALTERNATIVE_PATH $ALTERNATIVE_NAME $TARGET
    done)

update-alternatives --install ${ALTERNATIVES[@]}

# Verify key symlinks were created
for tool in clang clang++ clang-format llvm-ar llvm-config lld; do
    if [[ -e "/usr/bin/$tool-$VERSION" ]]; then
        # Verify symlink was created
        if [[ ! -e "/usr/bin/$tool" ]]; then
            echo "ERROR: $tool-$VERSION exists but $tool symlink was not created" >&2
            exit 1
        fi
    else
        echo "ERROR: Expected tool $tool-$VERSION does not exist" >&2
        exit 1
    fi
done

rm llvm.sh
