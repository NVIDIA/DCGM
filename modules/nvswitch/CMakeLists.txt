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
add_subdirectory(tests)

add_library(nvswitch_interface INTERFACE)
target_include_directories(nvswitch_interface INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(nvswitch_interface INTERFACE dcgm_interface modules_interface module_common_interface sdk_nscq_interface)

add_library(nvswitch_objects OBJECT)
target_link_libraries(nvswitch_objects PUBLIC nvswitch_interface)
target_sources(
    nvswitch_objects
    PRIVATE
    DcgmModuleNvSwitch.cpp
    DcgmModuleNvSwitch.h
    DcgmNvSwitchManager.cpp
    DcgmNvSwitchManager.h
    dcgm_nvswitch_structs.h
)

add_library(dcgmmodulenvswitch SHARED)
dcgm_macro_modules_link_options(dcgmmodulenvswitch)
target_compile_options(dcgmmodulenvswitch PRIVATE -fPIC)
target_link_libraries(
    dcgmmodulenvswitch
    PRIVATE
        nvswitch_objects
        modules_objects
        common_objects
        module_common_objects
        dcgm
        sdk_nscq_loader
)
update_lib_ver(dcgmmodulenvswitch)
