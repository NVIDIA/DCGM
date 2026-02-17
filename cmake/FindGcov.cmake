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

#[=======================================================================[.rst:
FindGcov
--------

Locate an installation of ``gcov`` - the GNU test coverage program.

Imported Targets
^^^^^^^^^^^^^^^^

``Gcov::gcov``
    An imported executable target referring to the ``gcov`` executable

Result Variables
^^^^^^^^^^^^^^^^

``Gcov_EXECUTABLE``
    The filepath to the ``gcov`` executable.

#]=======================================================================]

set(_Gcov_NAMES gcov)

if (CMAKE_CROSSCOMPILING AND DEFINED CMAKE_LIBRARY_ARCHITECTURE)
  list(PREPEND _Gcov_NAMES "${CMAKE_LIBRARY_ARCHITECTURE}-gcov")
endif()

get_property(_Gcov_ENABLED_LANGUAGES GLOBAL PROPERTY ENABLED_LANGUAGES)

foreach (_Gcov_ENABLED_LANGUAGE IN LISTS _Gcov_ENABLED_LANGUAGES)
  if (CMAKE_${_Gcov_ENABLED_LANGUAGE}_COMPILER_ID STREQUAL "GNU")
    cmake_path(GET CMAKE_${_Gcov_ENABLED_LANGUAGE}_COMPILER
        PARENT_PATH _Gcov_HINT)

    find_program(Gcov_EXECUTABLE
        NAMES ${_Gcov_NAMES}
        HINTS "${_Gcov_HINT}"
        NO_CMAKE_PATH
        NO_CMAKE_ENVIRONMENT_PATH
        NO_SYSTEM_ENVIRONMENT_PATH
        NO_CMAKE_SYSTEM_PATH)

    if(Gcov_EXECUTABLE)
      execute_process(
          COMMAND "${Gcov_EXECUTABLE}" "--version"
          OUTPUT_VARIABLE Gcov_VERSION)

      string(REGEX REPLACE
          "gcov [(]GCC[)] ([0-9]+[.][0-9]+[.][0-9]+).*"
          "\\1"
          Gcov_VERSION
          "${Gcov_VERSION}")
    endif()

    break()
  endif()
endforeach()

unset(_Gcov_HINT)
unset(_Gcov_ENABLED_LANGUAGE)
unset(_Gcov_ENABLED_LANGUAGES)
unset(_Gcov_NAMES)

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(Gcov
    REQUIRED_VARS Gcov_EXECUTABLE Gcov_VERSION
    VERSION_VAR Gcov_VERSION
    HANDLE_VERSION_RANGE)

if (Gcov_FOUND)
  add_executable(Gcov::gcov IMPORTED)
  set_property(TARGET Gcov::gcov PROPERTY IMPORTED_LOCATION "${Gcov_EXECUTABLE}")
endif()
