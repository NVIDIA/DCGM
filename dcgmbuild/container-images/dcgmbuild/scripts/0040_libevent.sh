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

mkdir --parents libevent
tar xf /tmp/downloads/libevent.tar.gz -C libevent --strip-components=1

# Libevent's CMake doesn't obey conventions, patch it
pushd libevent
cat <<EOF | patch CMakeLists.txt
diff --git a/CMakeLists.txt b/CMakeLists.txt
index 142d913..85fdba9 100644
--- a/CMakeLists.txt
+++ b/CMakeLists.txt
@@ -19,7 +19,7 @@
 #       start libevent.sln
 #

-cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
+cmake_minimum_required(VERSION 4.0 FATAL_ERROR)

 if (POLICY CMP0054)
     cmake_policy(SET CMP0054 NEW)
@@ -1453,7 +1453,7 @@ endif()
 #

 set(EVENT_INSTALL_CMAKE_DIR
-    "\${CMAKE_INSTALL_PREFIX}/lib/cmake/libevent")
+    "lib/cmake/libevent")

 export(PACKAGE libevent)


EOF
popd

cmake \
    -S libevent \
    -B build \
    -D CMAKE_POSITION_INDEPENDENT_CODE=ON \
    -D CMAKE_STAGING_PREFIX=$CMAKE_STAGING_PREFIX \
    -D EVENT__LIBRARY_TYPE=STATIC \
    -D EVENT__DISABLE_OPENSSL=ON
cmake --build build --target install

rm -rf libevent
