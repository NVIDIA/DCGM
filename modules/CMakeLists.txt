# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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
add_library(modules_interface INTERFACE)

macro(dcgm_macro_modules_link_options module)
  target_link_options(
    ${module}
    PRIVATE
    -Xlinker
    --version-script=${CMAKE_SOURCE_DIR}/modules/modules.linux_def
    )
endmacro()

target_link_libraries(modules_interface INTERFACE dcgm_interface common_protobuf_objects module_common_interface module_common_interface)
target_include_directories(modules_interface INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

add_library(modules_objects OBJECT)
target_link_libraries(modules_objects PUBLIC modules_interface common_protobuf_objects module_common_objects)
target_sources(modules_objects PRIVATE
    DcgmModuleApi.cpp
    DcgmModule.cpp
    core/DcgmModuleCore.cpp
    )
