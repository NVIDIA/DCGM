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
find_package(fmt REQUIRED)

add_library(modules_interface INTERFACE)

target_link_libraries(modules_interface
  INTERFACE
    dcgm_interface
    common_protobuf_interface
    module_common_interface
    module_common_interface
)
target_include_directories(modules_interface
  INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

add_library(modules_objects STATIC)
target_link_libraries(modules_objects
  PUBLIC
    modules_interface
    module_common_objects
)
target_sources(modules_objects
  PRIVATE
    DcgmModuleApi.cpp
    DcgmModule.cpp
    core/DcgmModuleCore.cpp
)

set(LOCAL_CURRENT_LIST_DIR ${CMAKE_CURRENT_SOURCE_DIR} CACHE INTERNAL "")

function(define_dcgm_module lib_name)
  target_link_libraries(${lib_name}
    PRIVATE
      modules_objects
      dcgm_common
      dcgm_logging
      dcgm_mutex
      dcgm
      fmt::fmt
  )
  target_compile_definitions(${lib_name} PRIVATE DCGM_API_EXPORT=1)
  target_link_options(${lib_name} PRIVATE -Wl,--version-script,${LOCAL_CURRENT_LIST_DIR}/dcgm_modules.linux_def
  )
endfunction()
