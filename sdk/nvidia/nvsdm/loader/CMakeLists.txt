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
target_include_directories(sdk_nvsdm_loader_interface
  INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(sdk_nvsdm_loader_interface INTERFACE sdk_nvsdm_interface)

target_sources(sdk_nvsdm_loader PRIVATE nvsdm_loader.cpp)
target_include_directories(sdk_nvsdm_loader
  PRIVATE
    ${NVSDM_INCLUDE_DIRS})

target_link_libraries(sdk_nvsdm_loader
  INTERFACE
    sdk_nvsdm_loader_interface
  PRIVATE
    sdk_nvsdm_interface
    dcgm_interface
    dl)
