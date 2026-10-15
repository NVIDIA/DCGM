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

#[=======================================================================[.rst:
Cuda
----

Cache the CUDA shared libraries DCGM links against dynamically. The build
environment, rather than the source tree, provides those libraries, so binaries
in the build tree cannot otherwise resolve them outside of the build container.

Every linked target is given a build tree RPATH entry naming the cache, so that
the copies are found ahead of the build environment. The entry is relative to
each binary, so the build tree survives relocation. The registration is
deferred until the end of top-level directory processing, so targets declared
in child directories are included.

Variables
^^^^^^^^^

``Cuda_LIBRARY_CACHE_DIR``
  Directory the CUDA shared libraries are copied to. Defaults to
  ``CMAKE_BINARY_DIR``.

Targets
^^^^^^^

The targets below are excluded from the default build. The libraries are large,
and only binaries that run outside of the build environment need them.

``cuda-library-cache``
  Caches every CUDA shared library.

``cuda-library-cache-<target>``
  Caches a single CUDA shared library, for example ``cuda-library-cache-cublas12``.
  Each copy is accompanied by the symbolic link through which the loader
  resolves the SONAME.

#]=======================================================================]

include_guard(GLOBAL)

find_package(Cuda REQUIRED)

set(Cuda_LIBRARY_CACHE_DIR "${CMAKE_BINARY_DIR}" CACHE PATH
    "Directory the CUDA shared libraries are copied to")

add_custom_target(cuda-library-cache)

foreach (CUDA_LIBRARY IN ITEMS
    cublas11
    cublasLt11
    cublas12
    cublasLt12
    cublas13
    cublasLt13)

    # Target-dependent generator expressions are not permitted in the OUTPUT
    # argument of add_custom_command(), so resolve both names now.
    get_target_property(CUDA_LIBRARY_LOCATION Cuda::${CUDA_LIBRARY} IMPORTED_LOCATION)
    get_target_property(CUDA_LIBRARY_SONAME Cuda::${CUDA_LIBRARY} IMPORTED_SONAME)
    file(REAL_PATH "${CUDA_LIBRARY_LOCATION}" CUDA_LIBRARY_REAL_PATH)
    cmake_path(GET CUDA_LIBRARY_REAL_PATH FILENAME CUDA_LIBRARY_FILENAME)

    set(CUDA_LIBRARY_CACHE_FILE
        "${Cuda_LIBRARY_CACHE_DIR}/${CUDA_LIBRARY_FILENAME}")

    add_custom_command(
        OUTPUT "${CUDA_LIBRARY_CACHE_FILE}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory
            "${Cuda_LIBRARY_CACHE_DIR}"
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            "${CUDA_LIBRARY_REAL_PATH}"
            "${CUDA_LIBRARY_CACHE_FILE}"
        DEPENDS "${CUDA_LIBRARY_REAL_PATH}"
        COMMENT "Copying ${CUDA_LIBRARY_FILENAME}"
        VERBATIM)

    add_custom_command(
        OUTPUT "${Cuda_LIBRARY_CACHE_DIR}/${CUDA_LIBRARY_SONAME}"
        COMMAND "${CMAKE_COMMAND}" -E create_symlink
            "${CUDA_LIBRARY_FILENAME}"
            "${Cuda_LIBRARY_CACHE_DIR}/${CUDA_LIBRARY_SONAME}"
        DEPENDS "${CUDA_LIBRARY_CACHE_FILE}"
        COMMENT "Linking ${CUDA_LIBRARY_SONAME}"
        VERBATIM)

    add_custom_target(cuda-library-cache-${CUDA_LIBRARY}
        DEPENDS "${Cuda_LIBRARY_CACHE_DIR}/${CUDA_LIBRARY_SONAME}")

    add_dependencies(cuda-library-cache cuda-library-cache-${CUDA_LIBRARY})
endforeach()

unset(CUDA_LIBRARY_CACHE_FILE)
unset(CUDA_LIBRARY_FILENAME)
unset(CUDA_LIBRARY_LOCATION)
unset(CUDA_LIBRARY_REAL_PATH)
unset(CUDA_LIBRARY_SONAME)

function(_Cuda_configure_build_rpath directory)
    get_property(targets DIRECTORY "${directory}" PROPERTY BUILDSYSTEM_TARGETS)

    foreach(target IN LISTS targets)
        get_target_property(type "${target}" TYPE)
        if(type STREQUAL "EXECUTABLE" OR type STREQUAL "SHARED_LIBRARY")
            set_property(TARGET "${target}" APPEND PROPERTY BUILD_RPATH
                "$ORIGIN/$<PATH:RELATIVE_PATH,${Cuda_LIBRARY_CACHE_DIR},$<TARGET_FILE_DIR:${target}>>")
        endif()
    endforeach()

    get_property(subdirs DIRECTORY "${directory}" PROPERTY SUBDIRECTORIES)
    foreach(subdir IN LISTS subdirs)
        _Cuda_configure_build_rpath("${subdir}")
    endforeach()
endfunction()

cmake_language(DEFER DIRECTORY "${CMAKE_SOURCE_DIR}"
  CALL _Cuda_configure_build_rpath "${CMAKE_SOURCE_DIR}")
