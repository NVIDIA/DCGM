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
add_library(config_interface INTERFACE)
target_include_directories(config_interface INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(config_interface INTERFACE common_interface dcgm_interface modules_interface module_common_interface)

add_library(config_objects OBJECT)
target_link_libraries(config_objects PUBLIC config_interface)
target_sources(
    config_objects
    PRIVATE
    DcgmModuleConfig.cpp
    dcgm_config_structs.h
    DcgmConfigManager.h
    DcgmModuleConfig.h
    DcgmConfigManager.cpp
)

add_library(dcgmmoduleconfig SHARED)
dcgm_macro_modules_link_options(dcgmmoduleconfig)
target_link_libraries(
    dcgmmoduleconfig
    PRIVATE
        config_objects
        modules_objects
        sdk_nvml_essentials_objects
        dcgm
        module_common_objects
)
update_lib_ver(dcgmmoduleconfig)
