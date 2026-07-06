#
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
#

#[[
Rust.cmake - CMake module for integrating Rust crates into the DCGM build.

Prerequisites:
  - Corrosion (https://github.com/corrosion-rs/corrosion) must be installed and findable
    via find_package(). Currently tested with Corrosion 0.5.x.
  - Rust toolchain (rustc, cargo) must be available in PATH.
  - For cross-compilation, the appropriate toolchain file must be used:
      cmake -DCMAKE_TOOLCHAIN_FILE=cmake/aarch64-linux-gnu-toolchain-gcc.cmake ..
      cmake -DCMAKE_TOOLCHAIN_FILE=cmake/x86_64-linux-gnu-toolchain-gcc.cmake ..

Supported architectures:
  - x86_64 (x86_64-unknown-linux-gnu)
  - aarch64 (aarch64-unknown-linux-gnu)

Public API:
  rust_crate()              - Import a Rust crate into the CMake build

Usage:
  include(Rust)
  rust_crate(
      MANIFEST_PATH path/to/Cargo.toml
      CRATE_NAME my_crate
      IMPORTED_CRATES my_imported_crates_var
  )

  # With custom environment variables for build.rs:
  rust_crate(
      MANIFEST_PATH path/to/Cargo.toml
      CRATE_NAME my_crate
      ENV "CUDA_PATH=/usr/local/cuda" "MY_VAR=value"
  )

Environment variables (read at configure time):
  CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER  - Override linker for x86_64 targets
  CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER - Override linker for aarch64 targets

Environment variables (set for build.rs scripts):
  CMAKE_OUT_DIR         - CMake binary directory for generated file output
  CMAKE_PROJECT_VERSION - Project version string from CMakeLists.txt
]]

include_guard(GLOBAL)

if(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
    message(STATUS "Rust target: x86_64-unknown-linux-gnu")
    set(Rust_CARGO_TARGET "x86_64-unknown-linux-gnu")
elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
    message(STATUS "Rust target: aarch64-unknown-linux-gnu")
    set(Rust_CARGO_TARGET "aarch64-unknown-linux-gnu")
else()
    message(STATUS "CMAKE_SYSTEM_PROCESSOR=${CMAKE_SYSTEM_PROCESSOR}")
    message(FATAL_ERROR "Unsupported architecture")
endif()

string(REPLACE "-" "_" _Rust_LINKER_ENV "${Rust_CARGO_TARGET}")
string(TOUPPER "CARGO_TARGET_${_Rust_LINKER_ENV}_LINKER" _Rust_LINKER_ENV)

set(_Rust_LINKER "${CMAKE_C_COMPILER}")
if (DEFINED ENV{${_Rust_LINKER_ENV}} AND NOT "$ENV{${_Rust_LINKER_ENV}}" STREQUAL "")
    set(_Rust_LINKER "$ENV{${_Rust_LINKER_ENV}}")
endif()
set(Rust_LINKER "${_Rust_LINKER}" CACHE FILEPATH "Linker used for Rust crates")

find_package(Corrosion REQUIRED)

# rust_crate
#
# Function to import a Rust crate into the CMake build.
#
# Parameters:
#   MANIFEST_PATH <Path/to/Cargo.toml>    - (Required) Path to the Cargo.toml manifest of the Rust crate.
#   CRATE_NAME <CrateName>                - (Required) Name of the single Rust crate target to import.
#   IMPORTED_CRATES <variable>            - (Optional) Variable name to receive the list of imported crate targets.
#   PROFILE <profile>                     - (Optional) Cargo build profile. Defaults to "release" for
#                                           Release/MinSizeRel/RelWithDebInfo configurations, "dev" otherwise.
#   ENV <KEY=VALUE>...                    - (Optional) List of "KEY=VALUE" environment variable pairs to set
#                                           for the crate's build.rs scripts, in addition to the default
#                                           CMAKE_OUT_DIR and CMAKE_PROJECT_VERSION.
#   [ARGS...]                             - (Optional) Additional arguments passed through to `corrosion_import_crate`.
#
# Target naming convention:
#   For each library crate, Corrosion creates up to three CMake targets:
#
#     <target>          INTERFACE library that acts as a facade. Consumers should
#                       target_link_libraries() against this name. It forwards to
#                       exactly one of the internal targets below.
#     <target>-static   STATIC IMPORTED library pointing to the Cargo staticlib artifact
#                       (.a/.lib). Also carries native link dependencies (e.g., -lpthread,
#                       -ldl) on INTERFACE_LINK_LIBRARIES. Only created when the crate
#                       produces a staticlib.
#     <target>-shared   SHARED IMPORTED library pointing to the Cargo cdylib artifact
#                       (.so/.dylib/.dll). Only created when the crate produces a cdylib.
#
#   Which inner target <target> forwards to depends on what the crate produces:
#     - staticlib only  -> <target> forwards to <target>-static  (always)
#     - cdylib only     -> <target> forwards to <target>-shared  (always)
#     - both            -> BUILD_SHARED_LIBS controls the selection:
#                            ON  -> <target> forwards to <target>-shared
#                            OFF -> <target> forwards to <target>-static
#
#   Important: BUILD_SHARED_LIBS only affects which inner target <target> resolves
#   to at link time. It does NOT change which targets Corrosion creates or which
#   artifacts Cargo builds -- both the .a and .so are always produced when the
#   Cargo.toml declares both crate types.
#
#   Consumers should link against the base <target> name. To force a specific linkage
#   type regardless of BUILD_SHARED_LIBS, link directly against <target>-shared or
#   <target>-static.
#
# This function performs the following:
#   - Imports the specified Rust crate using `corrosion_import_crate`, targeting the architecture
#     specified by Rust_CARGO_TARGET (set based on CMAKE_SYSTEM_PROCESSOR).
#   - Uses LOCKED and FROZEN flags for reproducible builds.
#   - Sets the `CMAKE_OUT_DIR` and `CMAKE_PROJECT_VERSION` environment variables for each imported
#     crate, so build.rs scripts can output generated files to the correct CMake binary directory
#     and access the project version.
#   - For shared library outputs (cdylib), sets the soname to
#     lib<crate_name>.so.<major_version> via rustc linker flags.
#   - Explicitly sets the linker for each imported crate, overriding Corrosion's default linker
#     selection logic. The linker is chosen as follows:
#       - If the architecture is x86_64 and the environment variable
#         CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER is set, use its value.
#       - If the architecture is aarch64 and the environment variable
#         CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER is set, use its value.
#       - Otherwise, fall back to using CMAKE_C_COMPILER.
#
# This ensures consistent and predictable integration of Rust crates into the CMake build,
# especially in cross-compilation or custom toolchain environments.
#
# Examples:
#   rust_crate(
#       MANIFEST_PATH cuda12/Cargo.toml
#       CRATE_NAME nvvs_rust_test_plugin_cuda12
#       IMPORTED_CRATES my_imported_crates
#   )
#
#   # With custom environment variables for build.rs:
#   rust_crate(
#       MANIFEST_PATH cuda12/Cargo.toml
#       CRATE_NAME nvvs_rust_test_plugin_cuda12
#       ENV "CUDA_PATH=/usr/local/cuda"
#           "MY_CUSTOM_VAR=some_value"
#   )
function(rust_crate)

    cmake_parse_arguments(PARSE_ARGV 0
        RUST                                                    # prefix
        ""                                                      # options (boolean flags)
        "MANIFEST_PATH;CRATE_NAME;IMPORTED_CRATES;PROFILE"      # one-value keywords
        "ENV"                                                    # multi-value keywords
    )

    if(RUST_KEYWORDS_MISSING_VALUES)
        message(FATAL_ERROR "rust_crate(): missing value(s) for keyword(s): ${RUST_KEYWORDS_MISSING_VALUES}")
    endif()

    foreach(RUST_KEY IN ITEMS RUST_MANIFEST_PATH RUST_CRATE_NAME)
        if(NOT DEFINED ${RUST_KEY} OR ${RUST_KEY} STREQUAL "")
            message(FATAL_ERROR "(${CMAKE_CURRENT_FUNCTION}) ${RUST_KEY} is required and cannot be empty")
        endif()
    endforeach()

    if(NOT DEFINED RUST_PROFILE)
        set(RUST_PROFILE "$<IF:$<CONFIG:Release,MinSizeRel,RelWithDebInfo>,release,dev>")
    endif()

    corrosion_import_crate(
        MANIFEST_PATH "${RUST_MANIFEST_PATH}"
        CRATES ${RUST_CRATE_NAME}
        CARGO_TARGET ${Rust_CARGO_TARGET}
        IMPORTED_CRATES _IMPORTED_CRATES
        PROFILE ${RUST_PROFILE}
        # LOCKED # Comment out LOCKED and FROZEN as our vendoring is still in progress.
        # FROZEN # Comment out LOCKED and FROZEN as our vendoring is still in progress.
        ${RUST_UNPARSED_ARGUMENTS}
    )

    foreach(imported_crate IN LISTS _IMPORTED_CRATES)
        # Set the CMAKE_OUT_DIR for rust crates so that build.rs scripts could understand where to put the
        # generated files.
        corrosion_set_env_vars(${imported_crate} "CMAKE_OUT_DIR=${CMAKE_CURRENT_BINARY_DIR}")
        corrosion_set_env_vars(${imported_crate} "CMAKE_PROJECT_VERSION=${CMAKE_PROJECT_VERSION}")
        # CMake's Unix Makefiles generator does not mark Corrosion's Cargo
        # custom targets as GNU make jobserver-aware. If Cargo inherits those
        # MAKEFLAGS, it can see stale jobserver file descriptors and fail with
        # "early EOF on jobserver pipe". Let Cargo create its own jobserver.
        corrosion_set_env_vars(${imported_crate} "CARGO_MAKEFLAGS=")
        corrosion_set_env_vars(${imported_crate} "MAKEFLAGS=")

        foreach(env_var IN LISTS RUST_ENV)
            corrosion_set_env_vars(${imported_crate} "${env_var}")
        endforeach()

        set(_variants "")

        if(TARGET ${imported_crate}-shared)
            list(APPEND _variants "shared")
            string(REPLACE "-" "_" _soname_crate "${imported_crate}")
            set(_soname_major_file "lib${_soname_crate}.so.${CMAKE_PROJECT_VERSION_MAJOR}")
            set(_soname_full_file "lib${_soname_crate}.so.${CMAKE_PROJECT_VERSION}")
            set(_cdylib_path "${CMAKE_CURRENT_BINARY_DIR}/lib${_soname_crate}.so")
            set(_soname_major_path "${CMAKE_CURRENT_BINARY_DIR}/${_soname_major_file}")
            set(_soname_full_path  "${CMAKE_CURRENT_BINARY_DIR}/${_soname_full_file}")

            corrosion_add_target_local_rustflags(${imported_crate}
                -Clink-arg=-Wl,-soname,${_soname_major_file})

            # Cargo emits exactly one file (lib<crate>.so) for a cdylib, while
            # NVVS's plugin loader (TestFramework::LoadPluginWithDir) only
            # discovers files matching *.so.<digits>, and the DCGM packaging
            # convention for C++ plugins ships the fully-versioned file
            # (lib<crate>.so.<MAJOR>.<MINOR>.<PATCH>) as the real binary with
            # lib<crate>.so.<MAJOR> as a soname symlink.
            #
            # Reshape Corrosion's copy of the cdylib to match that convention:
            #   lib<crate>.so           -> lib<crate>.so.<MAJOR>      (symlink)
            #   lib<crate>.so.<MAJOR>   -> lib<crate>.so.<FULL>       (symlink)
            #   lib<crate>.so.<FULL>                                  (real file)
            #
            # Corrosion attaches its own POST_BUILD copy_if_different to
            # _cargo-build_<crate> via cmake_language(DEFER), so our commands
            # must fire strictly after that copy lands the cdylib in
            # CMAKE_CURRENT_BINARY_DIR. Using a separate custom_target that
            # depends on _cargo-build_<crate> guarantees that ordering: CMake
            # runs the dependency's build (including all its POST_BUILD
            # commands — deferred or not) to completion before starting the
            # dependent target.
            # BYPRODUCTS intentionally omits ${_cdylib_path}: Corrosion's own
            # _cargo-build_<crate> rule already declares lib<crate>.so as a
            # ninja output (it copies the cdylib from the Cargo target dir
            # into CMAKE_CURRENT_BINARY_DIR). Listing it here too makes ninja
            # see two rules generating the same output and refuse to build
            # ("multiple rules generate ..."). The chain only re-points that
            # path as a symlink at the tail; the genuinely-new outputs are
            # the two version-suffixed paths.
            add_custom_target(${imported_crate}-soname-chain ALL
                COMMAND ${CMAKE_COMMAND} -E copy
                    "${_cdylib_path}"
                    "${_soname_full_path}.tmp"
                COMMAND ${CMAKE_COMMAND} -E rm -f
                    "${_cdylib_path}"
                    "${_soname_major_path}"
                    "${_soname_full_path}"
                COMMAND ${CMAKE_COMMAND} -E rename
                    "${_soname_full_path}.tmp"
                    "${_soname_full_path}"
                COMMAND ${CMAKE_COMMAND} -E create_symlink
                    "${_soname_full_file}"
                    "${_soname_major_path}"
                COMMAND ${CMAKE_COMMAND} -E create_symlink
                    "${_soname_major_file}"
                    "${_cdylib_path}"
                BYPRODUCTS
                    "${_soname_major_path}"
                    "${_soname_full_path}"
                VERBATIM
            )
            add_dependencies(${imported_crate}-soname-chain
                _cargo-build_${imported_crate})

            # install() uses FOLLOW_SYMLINK_CHAIN on this path, which picks up
            # the soname symlink and the real file it points at, but stops
            # short of the unversioned lib<crate>.so (that points *into* the
            # chain, not out of it) — matching NAMELINK_SKIP for C++ plugins.
            set_property(TARGET ${imported_crate}-shared PROPERTY
                RUST_SONAME_LINK "${_soname_major_path}")
        endif()
        if(TARGET ${imported_crate}-static)
            list(APPEND _variants "static")
        endif()

        # Keep Cargo's release profile for RelWithDebInfo, but explicitly restore Rust debug info.
        string(TOLOWER "${CMAKE_BUILD_TYPE}" _build_type_lower)
        if(_build_type_lower STREQUAL "relwithdebinfo")
            corrosion_add_target_local_rustflags(${imported_crate} "-Cdebuginfo=2")
        endif()

        # Corrosion has some logic to be "smart" about the used linker. Unfortunately, in our environment that creates
        # more problems than solutions. To disable the logic in corrosion-rs, we need to set the linker for each
        # imported crate manually.
        corrosion_set_linker(${imported_crate} "${Rust_LINKER}")

        if(NOT _variants)
            list(APPEND _variants "bin")
        endif()
        list(JOIN _variants ", " _variants_str)
        message(STATUS "Imported Rust Crate: ${imported_crate} [${_variants_str}]")

    endforeach()

    # Check if arguments contained IMPORTED_CRATES and set the corresponding variable for the caller
    if(DEFINED RUST_IMPORTED_CRATES AND NOT RUST_IMPORTED_CRATES STREQUAL "")
        set(${RUST_IMPORTED_CRATES} ${_IMPORTED_CRATES} PARENT_SCOPE)
    endif()

endfunction()
