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

BIN_DIR=/usr/bin

# Collect all versioned binaries and install them as alternatives
while IFS= read -r FILEPATH; do
    # Skip if not a file or not executable
    [[ -f "$FILEPATH" && -x "$FILEPATH" ]] || continue

    basename=$(basename "$FILEPATH")

    # Match files ending with -<version> where version is one or more digits
    # This handles: clang-21, clang++-21, clang-format-21, lld-21, llvm-ar-21, etc.
    if [[ "$basename" =~ ^(.+)-([0-9]+)$ ]]; then
        base_name="${BASH_REMATCH[1]}"
        version="${BASH_REMATCH[2]}"

        # Only process clang*, llvm*, or lld* related files
        if [[ "$base_name" =~ ^(clang|llvm|lld) ]]; then
            link_path="$BIN_DIR/$base_name"

            # Use version number as priority (higher version = higher priority)
            priority="$version"

            update-alternatives --install "$link_path" "$base_name" "$FILEPATH" "$priority"
            update-alternatives --set "$base_name" "$FILEPATH"
        fi
    fi
done < <(find ${BIN_DIR} -iname "*-$VERSION")

# Verify key symlinks were created
for tool in clang clang++ clang-format llvm-ar llvm-config lld; do
    if [[ -e "$BIN_DIR/$tool-$VERSION" ]]; then
        # Verify symlink was created
        if [[ ! -e "$BIN_DIR/$tool" ]]; then
            echo "ERROR: $tool-$VERSION exists but $tool symlink was not created" >&2
            exit 1
        fi
        echo "✓ Verified: $tool -> $tool-$VERSION"
    else
        echo "ERROR: Expected tool $tool-$VERSION does not exist" >&2
        exit 1
    fi
done

rm llvm.sh
