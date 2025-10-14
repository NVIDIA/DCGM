#
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

set(_Cuda_VERSIONS 11.8 12.9 13.0)

if (CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
    set(_Cuda_TARGET_DIR "targets/x86_64-linux")
elseif (CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
    set(_Cuda_TARGET_DIR "targets/sbsa-linux")
else()
    if (NOT Cuda_FIND_QUIETLY)
        message(WARNING
            "CMAKE_SYSTEM_PROCESSOR (${CMAKE_SYSTEM_PROCESSOR}) not supported by FindCuda module")
    endif()
    set(_Cuda_TARGET_DIR "targets/deadbeef-linux")
endif()

set(_Cuda_PATHS)
foreach (_Cuda_ROOT IN LISTS CMAKE_FIND_ROOT_PATH CMAKE_SYSROOT CMAKE_STAGING_PREFIX)
    list(APPEND _Cuda_PATHS "${_Cuda_ROOT}/usr/local")
endforeach()

unset(_Cuda_ROOT)

foreach (_Cuda_PATH IN ITEMS Cuda_ROOT CUDA_ROOT)
    if (DEFINED ${_Cuda_PATH})
        list(APPEND _Cuda_PATHS "${${_Cuda_PATH}}")
    endif()

    if (DEFINED ENV{${_Cuda_PATH}})
        list(APPEND _Cuda_PATHS "$ENV{${_Cuda_PATH}}")
    endif()
endforeach()

unset(_Cuda_PATH)

list(SORT _Cuda_PATHS)
list(REMOVE_DUPLICATES _Cuda_PATHS)

foreach(_Cuda_VERSION IN LISTS _Cuda_VERSIONS)
    set(_Cuda_FOUND TRUE)

    string(REGEX MATCH "^[^.]+" _Cuda_MAJOR_VERSION "${_Cuda_VERSION}")

    list(APPEND _Cuda_REQUIRED_VARS
        "CUDA${_Cuda_MAJOR_VERSION}_INCLUDE_DIR"
        "CUDA${_Cuda_MAJOR_VERSION}_STATIC_LIBS"
        "CUDA${_Cuda_MAJOR_VERSION}_LIBS"
        "CUDA${_Cuda_MAJOR_VERSION}_STATIC_CUBLAS_LIBS")

    list(TRANSFORM _Cuda_PATHS
        APPEND "/cuda-${_Cuda_VERSION}/${_Cuda_TARGET_DIR}/include"
        OUTPUT_VARIABLE _Cuda_INSTALL_ROOT_CANDIDATES)

    find_path(Cuda_${_Cuda_MAJOR_VERSION}_INCLUDE_DIR cuda.h
        PATHS ${_Cuda_INSTALL_ROOT_CANDIDATES}
        NO_SYSTEM_ENVIRONMENT_PATH)

    mark_as_advanced(Cuda_${_Cuda_MAJOR_VERSION}_INCLUDE_DIR)
    unset(_Cuda_INSTALL_ROOT_CANDIDATES)

    if (NOT Cuda_${_Cuda_MAJOR_VERSION}_INCLUDE_DIR)
      continue()
    endif ()

    cmake_path(GET Cuda_${_Cuda_MAJOR_VERSION}_INCLUDE_DIR
        PARENT_PATH _Cuda_INSTALL_ROOT)

    find_library(
        Cuda_${_Cuda_MAJOR_VERSION}_cuda_LIBRARY
        cuda
        HINTS "${_Cuda_INSTALL_ROOT}/lib"
        PATH_SUFFIXES stubs
        NO_PACKAGE_ROOT_PATH
        NO_CMAKE_PATH
        NO_CMAKE_ENVIRONMENT_PATH
        NO_SYSTEM_ENVIRONMENT_PATH
        NO_CMAKE_INSTALL_PREFIX
        NO_CMAKE_FIND_ROOT_PATH)

    mark_as_advanced(Cuda_${_Cuda_MAJOR_VERSION}_cuda_LIBRARY)

    if (NOT Cuda_${_Cuda_MAJOR_VERSION}_cuda_LIBRARY)
        set(_Cuda_FOUND FALSE)
    endif()

    find_library(
      Cuda_${_Cuda_MAJOR_VERSION}_culibos_LIBRARY
      culibos
      HINTS "${_Cuda_INSTALL_ROOT}/lib"
      NO_PACKAGE_ROOT_PATH
      NO_CMAKE_PATH
      NO_CMAKE_ENVIRONMENT_PATH
      NO_SYSTEM_ENVIRONMENT_PATH
      NO_CMAKE_INSTALL_PREFIX
      NO_CMAKE_FIND_ROOT_PATH)

    mark_as_advanced(Cuda_${_Cuda_MAJOR_VERSION}_culibos_LIBRARY)

    if (NOT Cuda_${_Cuda_MAJOR_VERSION}_culibos_LIBRARY)
        set(_Cuda_FOUND FALSE)
    endif()

    foreach(_Cuda_LIBRARY IN ITEMS cudart cublas cublasLt)
        find_library(
            Cuda_${_Cuda_MAJOR_VERSION}_${_Cuda_LIBRARY}_LIBRARY
            ${_Cuda_LIBRARY}_static
            HINTS "${_Cuda_INSTALL_ROOT}/lib"
            NO_PACKAGE_ROOT_PATH
            NO_CMAKE_PATH
            NO_CMAKE_ENVIRONMENT_PATH
            NO_SYSTEM_ENVIRONMENT_PATH
            NO_CMAKE_INSTALL_PREFIX
            NO_CMAKE_FIND_ROOT_PATH)

        mark_as_advanced(Cuda_${_Cuda_MAJOR_VERSION}_${_Cuda_LIBRARY}_LIBRARY)

        if (NOT Cuda_${_Cuda_MAJOR_VERSION}_${_Cuda_LIBRARY}_LIBRARY)
            set(_Cuda_FOUND FALSE)
        endif()
    endforeach()

    if (NOT _Cuda_FOUND)
        continue()
    endif()

    # TODO: Update Find module variables to conform to conventions.
    # https://cmake.org/cmake/help/latest/manual/cmake-developer.7.html#standard-variable-names
    set(CUDA${_Cuda_MAJOR_VERSION}_INCLUDE_DIR
        "${Cuda_${_Cuda_MAJOR_VERSION}_INCLUDE_DIR}")

    set(CUDA${_Cuda_MAJOR_VERSION}_STATIC_LIBS
        "${Cuda_${_Cuda_MAJOR_VERSION}_cudart_LIBRARY}"
        "${Cuda_${_Cuda_MAJOR_VERSION}_culibos_LIBRARY}")

    set(CUDA${_Cuda_MAJOR_VERSION}_LIBS "${Cuda_${_Cuda_MAJOR_VERSION}_cuda_LIBRARY}")
    set(CUDA${_Cuda_MAJOR_VERSION}_STATIC_CUBLAS_LIBS
        "${Cuda_${_Cuda_MAJOR_VERSION}_cublas_LIBRARY}"
        "${Cuda_${_Cuda_MAJOR_VERSION}_cublasLt_LIBRARY}")
endforeach()

unset(_Cuda_LIBRARY)
unset(_Cuda_FOUND)
unset(_Cuda_VERSION)
unset(_Cuda_VERSIONS)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Cuda REQUIRED_VARS ${_Cuda_REQUIRED_VARS})

if (NOT Cuda_FOUND)
    foreach(_Cuda_REQUIRED_VAR IN LISTS _Cuda_REQUIRED_VARS)
        unset(${_Cuda_REQUIRED_VAR})
    endforeach()
    unset(_Cuda_REQUIRED_VAR)
endif()

unset(_Cuda_REQUIRED_VARS)

# TODO: Define imported targets. Port call sites from module variables.
