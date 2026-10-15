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
    CRATE_NAME cpu_sdc
)

install(
    PROGRAMS "${CMAKE_CURRENT_BINARY_DIR}/mocked_cpu_sdc"
    DESTINATION "${DCGM_TESTS_INSTALL_DIR}/tests"
    COMPONENT Tests)

install(
    DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/res/"
    DESTINATION "${DCGM_TESTS_INSTALL_DIR}/tests/cpu_sdc_test_files"
    COMPONENT Tests
    FILES_MATCHING PATTERN "*.jsonl")
