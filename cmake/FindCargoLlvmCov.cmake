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

#[=======================================================================[.rst:
FindCargoLlvmCov
----------------

Locate an installation of ``cargo-llvm-cov`` - the cargo subcommand that wraps
``cargo test`` with LLVM source-based coverage instrumentation and exports
Cobertura/HTML/JSON reports.

The executable is a cargo subcommand, so it is invoked as
``cargo-llvm-cov llvm-cov <args>`` (the ``llvm-cov`` token is the subcommand
name cargo would otherwise pass).

Imported Targets
^^^^^^^^^^^^^^^^

``CargoLlvmCov::cargo-llvm-cov``
    An imported executable target referring to the ``cargo-llvm-cov`` executable

Result Variables
^^^^^^^^^^^^^^^^

``CargoLlvmCov_EXECUTABLE``
    The filepath to the ``cargo-llvm-cov`` executable.

``CargoLlvmCov_VERSION``
    The version of the ``cargo-llvm-cov`` executable, when it could be parsed
    from ``cargo-llvm-cov llvm-cov --version`` (a ``MAJOR.MINOR[.PATCH]``
    string). Informational only -- the module reports ``FOUND`` based on the
    executable, not the version.

#]=======================================================================]

find_program(CargoLlvmCov_EXECUTABLE cargo-llvm-cov)

# If it exists, extract its version. Note the `llvm-cov` subcommand token: a
# bare `cargo-llvm-cov --version` errors with "expected subcommand 'llvm-cov'".
if(CargoLlvmCov_EXECUTABLE)
  execute_process(
    COMMAND "${CargoLlvmCov_EXECUTABLE}" "llvm-cov" "--version"
    OUTPUT_VARIABLE _CargoLlvmCov_VERSION_OUTPUT
    ERROR_QUIET
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )

  if(_CargoLlvmCov_VERSION_OUTPUT MATCHES "cargo-llvm-cov ([0-9]+([.][0-9]+)+).*")
    set(CargoLlvmCov_VERSION "${CMAKE_MATCH_1}")
  endif()
endif()

# Execute standard find_package logic. Only the executable is REQUIRED: callers
# use this with find_package(CargoLlvmCov REQUIRED) to fail when the tool is
# absent, and that must hinge on the executable, not on parsing its --version.
# CargoLlvmCov_VERSION stays as VERSION_VAR (informational / version-range
# checks) so a present-but-unparseable version string does not hard-fail.
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CargoLlvmCov
    REQUIRED_VARS CargoLlvmCov_EXECUTABLE
    VERSION_VAR CargoLlvmCov_VERSION
    HANDLE_VERSION_RANGE
)

# Add target
if (CargoLlvmCov_FOUND AND NOT TARGET CargoLlvmCov::cargo-llvm-cov)
  add_executable(CargoLlvmCov::cargo-llvm-cov IMPORTED)
  set_property(TARGET CargoLlvmCov::cargo-llvm-cov PROPERTY IMPORTED_LOCATION "${CargoLlvmCov_EXECUTABLE}")
endif()

# Cleanup
unset(_CargoLlvmCov_VERSION_OUTPUT)
