#!/usr/bin/env sh

# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

set -eu

dependencies() {
    cat <<EOF
BOOST                          https://archives.boost.io/release/1.85.0/source/boost_1_85_0.tar.gz                                                                     sha256:be0d91732d5b0cc6fbb275c7939974457e79b54d6f07ce2e3dfdd68bef883b0b
CATCH2                         https://github.com/catchorg/Catch2/archive/refs/tags/v3.10.0.tar.gz                                                                     sha256:fc4303a5c2738beaa727066e126b5a28837a812230a3c5826caa38e7ab99ca48
CCACHE                         https://github.com/ccache/ccache/releases/download/v4.12.2/ccache-4.12.2-linux-x86_64.tar.xz                                            sha256:630c34ec94d451b200f5b14a6a25580d6a45bc80c394b7e0b93e33556eee5d32
CLANG                          https://apt.llvm.org/llvm.sh                                                                                                            sha256:03878e08f47b66cc95bc4b544b0db3c6d9ce8d60e6cf2492ae357984330a9eae
CMAKE                          https://github.com/Kitware/CMake/releases/download/v4.3.2/cmake-4.3.2-linux-x86_64.sh                                                   sha256:ee0b34a9a55a0d6220eceed0eab44047bbdbdc40fae0a89ba41635548a673fac
CORROSION                      https://github.com/corrosion-rs/corrosion/archive/refs/tags/v0.5.2.tar.gz                                                               sha256:6bc02411e29183a896aa60c58db6819ec6cf57c08997481d0b0da9029356b529
CROSSTOOL                      https://github.com/crosstool-ng/crosstool-ng/releases/download/crosstool-ng-1.28.0/crosstool-ng-1.28.0.tar.xz                           sha256:5750e29a2bda5cd8d67900592576b1670a1987a4dcd5e4f6beae09138a1f5699
CROSSTOOL_BINUTILS             https://mirror.cs.odu.edu/gnu/binutils/binutils-2.42.tar.xz                                                                                   sha256:f6e4d41fd5fc778b06b7891457b3620da5ecea1006c6a4a41ae998109f85a800
CROSSTOOL_GCC                  https://mirror.cs.odu.edu/gnu/gcc/gcc-14.3.0/gcc-14.3.0.tar.xz                                                                                sha256:e0dc77297625631ac8e50fa92fffefe899a4eb702592da5c32ef04e2293aca3a
CROSSTOOL_GDB                  https://mirror.cs.odu.edu/gnu/gdb/gdb-14.2.tar.xz                                                                                             sha256:2d4dd8061d8ded12b6c63f55e45344881e8226105f4d2a9b234040efa5ce7772
CROSSTOOL_GETTEXT              https://mirror.cs.odu.edu/gnu/gettext/gettext-0.22.5.tar.xz                                                                                   sha256:fe10c37353213d78a5b83d48af231e005c4da84db5ce88037d88355938259640
CROSSTOOL_GLIBC                https://mirror.cs.odu.edu/gnu/glibc/glibc-2.27.tar.xz                                                                                         sha256:5172de54318ec0b7f2735e5a91d908afe1c9ca291fec16b5374d9faadfc1fc72
CROSSTOOL_GMP                  https://gmplib.org/download/gmp/gmp-6.2.1.tar.xz                                                                                        sha256:fd4829912cddd12f84181c3451cc752be224643e87fac497b69edddadc49b4f2
CROSSTOOL_ISL                  https://libisl.sourceforge.io/isl-0.26.tar.xz                                                                                           sha256:a0b5cb06d24f9fa9e77b55fabbe9a3c94a336190345c2555f9915bb38e976504
CROSSTOOL_LIBICONV             https://mirror.cs.odu.edu/gnu/libiconv/libiconv-1.16.tar.gz                                                                             sha256:e6a1b1b589654277ee790cce3734f07876ac4ccfaecbee8afa0b649cf529cc04
CROSSTOOL_LINUX                https://cdn.kernel.org/pub/linux/kernel/v4.x/linux-4.20.17.tar.xz                                                                       sha256:d011245629b980d4c15febf080b54804aaf215167b514a3577feddb2495f8a3e
CROSSTOOL_MOLD                 https://github.com/rui314/mold/archive/refs/tags/v2.40.4.tar.gz                                                                         sha256:69414c702ec1084e1fa8ca16da24f167f549e5e11e9ecd5d70a8dcda6f08c249
CROSSTOOL_MPC                  https://www.multiprecision.org/downloads/mpc-1.3.1.tar.gz                                                                               sha256:ab642492f5cf882b74aa0cb730cd410a81edcdbec895183ce930e706c1c759b8
CROSSTOOL_MPFR                 https://www.mpfr.org/mpfr-4.2.2/mpfr-4.2.2.tar.xz                                                                                       sha256:b67ba0383ef7e8a8563734e2e889ef5ec3c3b898a01d00fa0a6869ad81c6ce01
CROSSTOOL_NCURSES              https://invisible-mirror.net/archives/ncurses/ncurses-6.4.tar.gz                                                                        sha256:6931283d9ac87c5073f30b6290c4c75f21632bb4fc3603ac8100812bed248159
CROSSTOOL_ZLIB                 https://github.com/madler/zlib/releases/download/v1.3.1/zlib-1.3.1.tar.xz                                                               sha256:38ef96b8dfe510d42707d9c781877914792541133e1870841463bfa73f883e32
CROSSTOOL_ZSTD                 https://github.com/facebook/zstd/releases/download/v1.5.5/zstd-1.5.5.tar.gz                                                             sha256:9c4396cc829cfae319a6e2615202e82aad41372073482fce286fac78646d3ee4
CUDA11_AARCH64_LINUX_GNU       https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_arm64.deb sha256:50d39ebfd4f7e05d191f65c26e4aac79d5f2ef68da515e71d0d83ddd38151534
CUDA11_CROSS_AARCH64_LINUX_GNU https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-cross-sbsa-ubuntu2004-11-8-local_11.8.0-1_all.deb  sha256:5381fb095285ee2f0bf6bab12d2da8997e24e4ba536a2aab944835508d1956b5
CUDA11_X86_64_LINUX_GNU        https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb sha256:94c5ab8de3d38ecca4608fdcdf07ef05d1459cf1a8871d65d4ae92621afef9e4
CUDA12_AARCH64_LINUX_GNU       https://developer.download.nvidia.com/compute/cuda/12.9.2/local_installers/cuda-repo-ubuntu2404-12-9-local_12.9.2-575.57.08-1_arm64.deb sha256:6633a840b545c86fa0c45e99a3fba7a2afdecd926b2c3dc7c5887767ada3a26e
CUDA12_CROSS_AARCH64_LINUX_GNU https://developer.download.nvidia.com/compute/cuda/12.9.2/local_installers/cuda-repo-cross-sbsa-ubuntu2404-12-9-local_12.9.2-1_all.deb  sha256:459585066f72244cdc4939c1141b73bf0c5151750e76094e39734d0a8024e014
CUDA12_X86_64_LINUX_GNU        https://developer.download.nvidia.com/compute/cuda/12.9.2/local_installers/cuda-repo-ubuntu2404-12-9-local_12.9.2-575.57.08-1_amd64.deb sha256:2713c8d6f8e88d79239578571218f0b3ba742fd2f88c9066445f1132ed57ace9
CUDA13_AARCH64_LINUX_GNU       https://developer.download.nvidia.com/compute/cuda/13.2.1/local_installers/cuda-repo-ubuntu2404-13-2-local_13.2.1-595.58.03-1_arm64.deb sha256:2969a40f34963e99afbfe29c680d09477ed2afc595e26a082488673a9e8c11ee
CUDA13_CROSS_AARCH64_LINUX_GNU https://developer.download.nvidia.com/compute/cuda/13.2.1/local_installers/cuda-repo-cross-sbsa-ubuntu2404-13-2-local_13.2.1-1_all.deb  sha256:8c9eb3b43fe0f99f24e1fff812b367209768317c4e03de5880e9fdc8efa0ae2b
CUDA13_X86_64_LINUX_GNU        https://developer.download.nvidia.com/compute/cuda/13.2.1/local_installers/cuda-repo-ubuntu2404-13-2-local_13.2.1-595.58.03-1_amd64.deb sha256:8841f965fcd24c6cfaa6cdf18ea5ee4956982063eaac6c85bbc54c4844d35978
FMT                            https://github.com/fmtlib/fmt/archive/refs/tags/12.0.0.tar.gz                                                                           sha256:aa3e8fbb6a0066c03454434add1f1fc23299e85758ceec0d7d2d974431481e40
GIT                            https://mirrors.edge.kernel.org/pub/software/scm/git/git-2.54.0.tar.xz                                                                  sha256:f689162364c10de79ef89aa8dbf48731eb057e34edbbd20aca510ce0154681a3
GIT_LFS                        https://github.com/git-lfs/git-lfs/releases/download/v3.7.1/git-lfs-linux-amd64-v3.7.1.tar.gz                                           sha256:1c0b6ee5200ca708c5cebebb18fdeb0e1c98f1af5c1a9cba205a4c0ab5a5ec08
JSONCPP                        https://github.com/open-source-parsers/jsoncpp/archive/1.9.6.tar.gz                                                                     sha256:f93b6dd7ce796b13d02c108bc9f79812245a82e577581c4c9aabe57075c90ea2
LCOV                           https://github.com/linux-test-project/lcov/archive/v2.0.tar.gz                                                                          sha256:3ec0795f2776e69e4c8a35a34ddd199d0ee2119a2e65a3a944374e52edb06b26
LIBEVENT                       https://github.com/libevent/libevent/releases/download/release-2.1.13-stable/libevent-2.1.13-stable.tar.gz                              sha256:f7e9383b8c0baa81b687e5b5eecc01beefaf1b19b64151d95ed61647fe7a315c
LIBNUMA                        https://github.com/numactl/numactl/archive/refs/tags/v2.0.16.tar.gz                                                                     sha256:a35c3bdb3efab5c65927e0de5703227760b1101f5e27ab741d8f32b3d5f0a44c
PLOG                           https://github.com/SergiusTheBest/plog/archive/1.1.10.tar.gz                                                                            sha256:55a090fc2b46ab44d0dde562a91fe5fc15445a3caedfaedda89fe3925da4705a
RIPGREP                        https://github.com/BurntSushi/ripgrep/releases/download/14.1.0/ripgrep-14.1.0-x86_64-unknown-linux-musl.tar.gz                          sha256:f84757b07f425fe5cf11d87df6644691c644a5cd2348a2c670894272999d3ba7
RUST                           https://static.rust-lang.org/dist/2026-07-16/rustc-1.97.1-src.tar.xz                                                                    sha256:0ed06fdaffd4722a7702e0b4eebfafc897ab8f513e8e1b247cdd7e5c6df6ded2
SCCACHE                        https://github.com/mozilla/sccache/releases/download/v0.14.0/sccache-v0.14.0-x86_64-unknown-linux-musl.tar.gz                           sha256:8424b38cda4ecce616a1557d81328f3d7c96503a171eab79942fad618b42af44
TCLAP                          https://git.code.sf.net/p/tclap/code.git                                                                                                1.4.0-rc2
YAML_CPP                       https://github.com/jbeder/yaml-cpp/archive/refs/tags/0.8.0.tar.gz                                                                       sha256:fbe74bbdcee21d656715688706da3c8becfd946d92cd44705cc6098bb23b3a16
ZLIB                           https://github.com/madler/zlib/releases/download/v1.3.1/zlib-1.3.1.tar.gz                                                               sha256:9a93b2b7dfdac77ceba5a558a580e74667dd6fede4585b91eefb60f03b72df23
EOF
}

# Usage
case "${1:-}" in
    bake | buildkit)
        MODE=$1
        ;;
    *)
        printf 'Usage: %s bake|buildkit\n' "$0" >&2
        exit 2
        ;;
esac

# Formatting
print_arg() {
    ARG_NAME=$1
    ARG_VALUE=$2

    case "$MODE" in
        bake)
            printf '%s\n' "--set=*.args.${ARG_NAME}=${ARG_VALUE}"
            ;;
        buildkit)
            printf '%s %s\n' "--opt" "build-arg:${ARG_NAME}=${ARG_VALUE}"
            ;;
    esac
}

# Transform dependency information into flags
dependencies | while read -r name url sha
do
    # Special cases
    if [ "$name" = "TCLAP" ]
    then
        print_arg "${name}_URL" "$url"
        print_arg "${name}_TAG" "$sha"
    # Everybody else
    else
        print_arg "${name}_URL" "$url"
        print_arg "${name}_SHA256SUM" "$sha"
    fi
done
