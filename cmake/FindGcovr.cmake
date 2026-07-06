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
FindGcovr
---------

Locate an installation of ``gcovr`` - a Python module for interacting with
the output of running executables compiled with support for ``gcov``

Imported Targets
^^^^^^^^^^^^^^^^

``Gcovr::gcovr``
    An imported executable target referring to the ``gcovr`` executable

Result Variables
^^^^^^^^^^^^^^^^

``Gcovr_EXECUTABLE``
    The filepath to the ``gcovr`` executable.

#]=======================================================================]

find_program(Gcovr_EXECUTABLE gcovr)

# If gcovr exists on the system, extract its version
if(Gcovr_EXECUTABLE)
  execute_process(
    COMMAND "${Gcovr_EXECUTABLE}" "--version"
    OUTPUT_VARIABLE _Gcovr_VERSION_OUTPUT
  )

  if(_Gcovr_VERSION_OUTPUT MATCHES "gcovr ([0-9]+([.][0-9]+)+).*")
    set(Gcovr_VERSION "${CMAKE_MATCH_1}")
  endif()
endif()

# Execute standard find_package logic
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Gcovr
    REQUIRED_VARS Gcovr_EXECUTABLE Gcovr_VERSION
    VERSION_VAR Gcovr_VERSION
    HANDLE_VERSION_RANGE
)

# Add target
if (Gcovr_FOUND AND NOT TARGET Gcovr::gcovr)
  add_executable(Gcovr::gcovr IMPORTED)
  set_property(TARGET Gcovr::gcovr PROPERTY IMPORTED_LOCATION "${Gcovr_EXECUTABLE}")
endif()

# Cleanup
unset(_Gcovr_VERSION_OUTPUT)
