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

if(BUILD_TESTING)
  add_library(hostengine_lib STATIC)
  target_sources(hostengine_lib
                 PRIVATE
                 src/HostEngineCommandLine.cpp
                 src/HostEngineMainHelpers.cpp
                 src/HostEngineOutput.cpp)
  target_include_directories(hostengine_lib PUBLIC src)
  target_link_libraries(hostengine_lib PRIVATE common_interface TCLAP::TCLAP)

  add_subdirectory(src/tests)
endif()

target_sources(nv-hostengine
   PRIVATE
       src/HostEngineCommandLine.cpp
       src/HostEngineCommandLine.h
       src/HostEngineMainHelpers.cpp
       src/HostEngineMainHelpers.h
       src/HostEngineOutput.cpp
       src/HostEngineOutput.h
       src/main.cpp)

target_link_libraries(nv-hostengine
    PRIVATE
        common_interface
        dcgm_interface
        dcgm
        buildinfo_objects
        dcgm_common
        TCLAP::TCLAP
        transport_objects
        dcgm_logging
    PUBLIC
        libevent::extra
        libevent::pthreads)

target_link_options(nv-hostengine PRIVATE -Wl,--version-script,${CMAKE_CURRENT_SOURCE_DIR}/hostengine.linux_def)
