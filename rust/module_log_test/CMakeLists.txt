# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

include(Rust)

rust_crate(
    MANIFEST_PATH ${CMAKE_CURRENT_SOURCE_DIR}/Cargo.toml
    CRATE_NAME module_log_test
)

install(CODE "
    file(INSTALL
        DESTINATION \"$<INSTALL_PREFIX>/${DCGM_TESTS_APP_DIR}/${DCGM_TESTS_ARCH}\"
        TYPE SHARED_LIBRARY
        FOLLOW_SYMLINK_CHAIN
        FILES \"${CMAKE_CURRENT_BINARY_DIR}/libplugin_log_test.so\")
    "
    COMPONENT Tests)

install(CODE "
    file(INSTALL
        DESTINATION \"$<INSTALL_PREFIX>/${DCGM_TESTS_APP_DIR}/${DCGM_TESTS_ARCH}\"
        TYPE EXECUTABLE
        FILES \"${CMAKE_CURRENT_BINARY_DIR}/module_log_test_host\")
    "
    COMPONENT Tests)
