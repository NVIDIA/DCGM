# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
add_executable(nv-hostengine)

target_sources(nv-hostengine
               PRIVATE
               src/HostEngineCommandLine.cpp
               src/HostEngineCommandLine.h
               src/HostEngineOutput.cpp
               src/HostEngineOutput.h
               src/main.cpp)

target_link_libraries(nv-hostengine PRIVATE common_interface dcgm_interface dcgm)
target_link_libraries(nv-hostengine PRIVATE buildinfo_objects dcgm_common transport_objects)

find_package(Libevent REQUIRED)
target_link_libraries(nv-hostengine PUBLIC ${LIBEVENT_STATIC_LIBS})

target_link_options(nv-hostengine PRIVATE -Wl,--version-script,${CMAKE_CURRENT_SOURCE_DIR}/hostengine.linux_def)
