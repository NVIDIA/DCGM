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
# Extractor script driven by `dcgm-bindgen`'s build-time fallback path
# (`CmakeToolchain::from_cmake_extract`). The Rust caller writes this file
# into `OUT_DIR` after replacing `@TOOLCHAIN_FILE@` with an absolute path to
# the real cross toolchain (`/opt/cross/<arch>/toolchain-gcc.cmake` in the
# build container) and then runs `cmake -P` on the result.
#
# The output contract: a single JSON object printed between the two marker
# lines `___DCGM_TOOLCHAIN_JSON_BEGIN___` and `___DCGM_TOOLCHAIN_JSON_END___`.
# Anything else (CMake's own startup chatter, deprecation warnings, etc.) is
# ignored by the Rust parser.

cmake_minimum_required(VERSION 3.20)

include("@TOOLCHAIN_FILE@")

# JSON-escape the minimum set of characters needed before wrapping a string
# in JSON quotes. `string(JSON ... SET)` requires its <value> argument to be
# valid JSON syntax, so callers must produce `"..."` with internal quotes
# and backslashes already escaped.
function(_dcgm_json_escape out_var input)
    string(REPLACE "\\" "\\\\" tmp "${input}")
    string(REPLACE "\"" "\\\"" tmp "${tmp}")
    string(REPLACE "\n" "\\n" tmp "${tmp}")
    string(REPLACE "\r" "\\r" tmp "${tmp}")
    string(REPLACE "\t" "\\t" tmp "${tmp}")
    set(${out_var} "${tmp}" PARENT_SCOPE)
endfunction()

# Build a JSON array from a CMake list variable. `list_var` is the *name* of
# the variable, not its contents -- the function dereferences it via
# `IN LISTS` so embedded semicolons split into elements.
function(_dcgm_json_array_from_list out_var list_var)
    set(_arr "[]")
    set(_idx 0)
    foreach(_item IN LISTS ${list_var})
        _dcgm_json_escape(_esc "${_item}")
        string(JSON _arr SET "${_arr}" ${_idx} "\"${_esc}\"")
        math(EXPR _idx "${_idx} + 1")
    endforeach()
    set(${out_var} "${_arr}" PARENT_SCOPE)
endfunction()

foreach(_required IN ITEMS
    CMAKE_SYSROOT
    CMAKE_LIBRARY_ARCHITECTURE
    CMAKE_C_STANDARD_INCLUDE_DIRECTORIES
    CMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES)
    if(NOT DEFINED ${_required})
        message(FATAL_ERROR
            "dcgm-bindgen: ${_required} is not defined after including the toolchain file")
    endif()
endforeach()

_dcgm_json_escape(_sysroot_esc       "${CMAKE_SYSROOT}")
_dcgm_json_escape(_target_triple_esc "${CMAKE_LIBRARY_ARCHITECTURE}")
_dcgm_json_array_from_list(_c_arr    CMAKE_C_STANDARD_INCLUDE_DIRECTORIES)
_dcgm_json_array_from_list(_cxx_arr  CMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES)

set(_json "{}")
string(JSON _json SET "${_json}" "sysroot"          "\"${_sysroot_esc}\"")
string(JSON _json SET "${_json}" "target_triple"    "\"${_target_triple_esc}\"")
string(JSON _json SET "${_json}" "c_include_dirs"   "${_c_arr}")
string(JSON _json SET "${_json}" "cxx_include_dirs" "${_cxx_arr}")

# CMake's `message()` writes to stderr by default, which is what the Rust
# parser reads. Sentinel markers make extraction robust against unrelated
# stderr noise (CMake warnings, toolchain `message(STATUS ...)` calls).
message("___DCGM_TOOLCHAIN_JSON_BEGIN___")
message("${_json}")
message("___DCGM_TOOLCHAIN_JSON_END___")
