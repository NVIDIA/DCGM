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

set_property(DIRECTORY "${CMAKE_SOURCE_DIR}" APPEND PROPERTY
  ADDITIONAL_CLEAN_FILES "${CMAKE_BINARY_DIR}/coverage")

include(CoverageCleanup)

find_package(Gcov)

if (Gcov_FOUND)
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

  if (Gcovr_FOUND)
    string(REGEX REPLACE "(.)" "\\\\\\1" CMAKE_SOURCE_DIR_PATTERN "${CMAKE_SOURCE_DIR}")

    add_custom_target(gcovr-cobertura
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_BINARY_DIR}/coverage/gcovr"
      COMMAND
        Gcovr::gcovr
        --exclude-unreachable-branches
        --filter "${CMAKE_SOURCE_DIR_PATTERN}/.*"
        --gcov-executable "${Gcov_EXECUTABLE}"
        --gcov-ignore-parse-errors
        --cobertura "${CMAKE_BINARY_DIR}/coverage/gcovr/coverage.xml"
        --cobertura-pretty
        --merge-mode-functions separate
        --root "${CMAKE_SOURCE_DIR}"
      COMMAND
        "${CMAKE_COMMAND}" -E echo
        "Gcovr-format C/C++ test coverage report available at: ${CMAKE_BINARY_DIR}/coverage/gcovr/coverage.xml"
      COMMENT "Generating Cobertura-format C/C++ coverage report...")

    add_custom_target(gcovr-html
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_BINARY_DIR}/coverage/gcovr/html"
      COMMAND
        Gcovr::gcovr
        --cobertura-add-tracefile "${CMAKE_BINARY_DIR}/coverage/gcovr/coverage.xml"
        --html-details  "${CMAKE_BINARY_DIR}/coverage/gcovr/html/index.html"
      COMMAND
        "${CMAKE_COMMAND}" -E echo "C/C++ HTML-format test coverage report available at: ${CMAKE_BINARY_DIR}/coverage/gcovr/html/index.html"
      COMMENT "Generating HTML-format C/C++ coverage report")

    add_dependencies(gcovr-html gcovr-cobertura)
  endif()
endif()

find_package(CargoLlvmCov)

if (CargoLlvmCov_FOUND)
  set(_CargoLlvmCov_TARGET_DIR
    "${CMAKE_BINARY_DIR}/coverage/cargo-llvm-cov/target")

  add_custom_target(cargo-llvm-cov-profile
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_BINARY_DIR}/coverage/cargo-llvm-cov/html"
    COMMAND
      "${CMAKE_COMMAND}" -E env CARGO_MAKEFLAGS=
      "CARGO_TARGET_DIR=${_CargoLlvmCov_TARGET_DIR}" MAKEFLAGS=
      $<TARGET_FILE:CargoLlvmCov::cargo-llvm-cov> llvm-cov clean --workspace
      --manifest-path "${CMAKE_SOURCE_DIR}/Cargo.toml"
    COMMAND
      "${CMAKE_COMMAND}" -E env CARGO_MAKEFLAGS=
      "CARGO_TARGET_DIR=${_CargoLlvmCov_TARGET_DIR}" MAKEFLAGS=
      $<TARGET_FILE:CargoLlvmCov::cargo-llvm-cov> llvm-cov test
      --workspace
      --locked
      --no-fail-fast
      --no-report
      --profile coverage
      --manifest-path "${CMAKE_SOURCE_DIR}/Cargo.toml"
    COMMENT "Collecting Rust test coverage profile")

  add_custom_target(cargo-llvm-cov-cobertura
    COMMAND
      "${CMAKE_COMMAND}" -E env CARGO_MAKEFLAGS=
      "CARGO_TARGET_DIR=${_CargoLlvmCov_TARGET_DIR}" MAKEFLAGS=
      $<TARGET_FILE:CargoLlvmCov::cargo-llvm-cov> llvm-cov report
      --cobertura
      --output-path "${CMAKE_BINARY_DIR}/coverage/cargo-llvm-cov/coverage.xml"
      --profile coverage
      --manifest-path "${CMAKE_SOURCE_DIR}/Cargo.toml"
      --ignore-filename-regex ".*/[.]cargo/registry/.*|.*/_out/build/.*"
    COMMAND
      "${CMAKE_COMMAND}" -E echo "Cobertura-format Rust test coverage report available at: ${CMAKE_BINARY_DIR}/coverage/cargo-llvm-cov/coverage.xml"
    COMMENT "Generating Cobertura-format Rust test coverage report"
    VERBATIM)

  add_dependencies(cargo-llvm-cov-cobertura cargo-llvm-cov-profile)

  add_custom_target(cargo-llvm-cov-html
    COMMAND
      "${CMAKE_COMMAND}" -E env CARGO_MAKEFLAGS=
      "CARGO_TARGET_DIR=${_CargoLlvmCov_TARGET_DIR}" MAKEFLAGS=
      $<TARGET_FILE:CargoLlvmCov::cargo-llvm-cov> llvm-cov report
      --html
      --output-dir "${CMAKE_BINARY_DIR}/coverage/cargo-llvm-cov"
      --profile coverage
      --manifest-path "${CMAKE_SOURCE_DIR}/Cargo.toml"
      --ignore-filename-regex ".*/[.]cargo/registry/.*|.*/_out/build/.*"
    COMMAND
      "${CMAKE_COMMAND}" -E echo "HTML-format Rust test coverage report available at: ${CMAKE_BINARY_DIR}/coverage/cargo-llvm-cov/html/index.html"
    COMMENT "Generating HTML-format Rust test coverage report"
    VERBATIM)

  add_dependencies(cargo-llvm-cov-html cargo-llvm-cov-profile)

  unset(_CargoLlvmCov_TARGET_DIR)
endif()

if (Gcovr_FOUND AND CargoLlvmCov_FOUND)
  add_custom_target(coverage-cobertura
    COMMAND
      Gcovr::gcovr
      --cobertura-add-tracefile "${CMAKE_BINARY_DIR}/coverage/gcovr/coverage.xml"
      --cobertura-add-tracefile "${CMAKE_BINARY_DIR}/coverage/cargo-llvm-cov/coverage.xml"
      --cobertura "${CMAKE_BINARY_DIR}/coverage/coverage.xml"
      --cobertura-pretty
      --root "${CMAKE_SOURCE_DIR}"
    COMMAND
      "${CMAKE_COMMAND}" -E echo "Cobertura-format merged test coverage report available at: ${CMAKE_BINARY_DIR}/coverage/coverage.xml"
    COMMENT "Generating Cobertura-format merged C/C++ and Rust test coverage report")

  add_dependencies(coverage-cobertura gcovr-cobertura cargo-llvm-cov-cobertura)

  add_custom_target(coverage-html
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_BINARY_DIR}/coverage/html"
    COMMAND
      Gcovr::gcovr
      --cobertura-add-tracefile "${CMAKE_BINARY_DIR}/coverage/gcovr/coverage.xml"
      --cobertura-add-tracefile "${CMAKE_BINARY_DIR}/coverage/cargo-llvm-cov/coverage.xml"
      --html-details  "${CMAKE_BINARY_DIR}/coverage/html/index.html"
      --print-summary
      --root "${CMAKE_SOURCE_DIR}"
    COMMAND
      "${CMAKE_COMMAND}" -E echo "HTML-format merged test coverage report available at: ${CMAKE_BINARY_DIR}/coverage/html/index.html"
    COMMENT "Generating HTML-format merged C/C++ and Rust test coverage report")

  add_dependencies(coverage-html gcovr-cobertura cargo-llvm-cov-cobertura)
endif()
