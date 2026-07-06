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

if (NOT DCGM_BUILD_COVERAGE)
  return()
endif()

find_package(Gcov)
if (NOT Gcov_FOUND)
  return()
endif()

cmake_path(GET Gcov_EXECUTABLE PARENT_PATH TOOLCHAIN_ROOT)
cmake_path(GET TOOLCHAIN_ROOT PARENT_PATH TOOLCHAIN_ROOT)
string(REGEX REPLACE "(.)" "[\\1]" TOOLCHAIN_ROOT "${TOOLCHAIN_ROOT}")

set(COVERAGE_COMMAND "${Gcov_EXECUTABLE}" CACHE FILEPATH
    "Path to the coverage program that CTest uses for performing coverage inspection.")

string(JOIN "|" CTEST_CUSTOM_COVERAGE_EXCLUDE
  "${TOOLCHAIN_ROOT}/.*"
  "/usr/.*"
  ".*sdk/.*"
  ".*sdk_samples/.*"
  ".*/test_common/.*"
  ".*/tests/.*"
  ".*testing/.*")

configure_file(cmake/CTestCustom.cmake.in CTestCustom.cmake @ONLY)

find_package(Gcovr)
if (NOT Gcovr_FOUND)
  return()
endif()

file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/coverage/gcovr/html")

string(REGEX REPLACE "(.)" "\\\\\\1"
  CMAKE_SOURCE_DIR_PATTERN
  "${CMAKE_SOURCE_DIR}")

add_custom_target(gcovr-cobertura
  COMMAND
    Gcovr::gcovr
    --cobertura "${CMAKE_BINARY_DIR}/coverage/gcovr/coverage.xml"
    --cobertura-pretty
    --exclude-unreachable-branches
    --filter "${CMAKE_SOURCE_DIR_PATTERN}/.*"
    --gcov-executable "${Gcov_EXECUTABLE}"
    --gcov-ignore-parse-errors
    --merge-mode-functions separate
    --root "${CMAKE_SOURCE_DIR}"
  COMMENT "Generating Cobertura-format coverage report")

add_custom_target(gcovr-html
  COMMAND
    Gcovr::gcovr
    --exclude-unreachable-branches
    --filter "${CMAKE_SOURCE_DIR_PATTERN}/.*"
    --gcov-executable "${Gcov_EXECUTABLE}"
    --gcov-ignore-parse-errors
    --html-details  "${CMAKE_BINARY_DIR}/coverage/gcovr/html/index.html"
    --merge-mode-functions separate
    --print-summary
    --root "${CMAKE_SOURCE_DIR}"
  COMMENT "Generating HTML coverage report")

add_custom_target(gcovr-json
  COMMAND
    Gcovr::gcovr
    --exclude-unreachable-branches
    --filter "${CMAKE_SOURCE_DIR_PATTERN}/.*"
    --gcov-executable "${Gcov_EXECUTABLE}"
    --gcov-ignore-parse-errors
    --json  "${CMAKE_BINARY_DIR}/coverage/gcovr/coverage.json"
    --merge-mode-functions separate
    --root "${CMAKE_SOURCE_DIR}"
  COMMENT "Generating Gcovr-format JSON coverage report")
