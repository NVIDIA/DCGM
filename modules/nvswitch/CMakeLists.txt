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

add_library(nvswitch_interface INTERFACE)
target_include_directories(nvswitch_interface INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(nvswitch_interface
    INTERFACE
        dcgm_interface
        common_interface
        modules_interface
        module_common_interface
        sdk_nscq_interface
        transport_interface
)

add_library(nvswitch_objects STATIC)
target_link_libraries(nvswitch_objects
    PRIVATE
        nvswitch_interface
)

set(SRCS 
    DcgmModuleNvSwitch.cpp
    DcgmModuleNvSwitch.h
    DcgmNvSwitchManager.cpp
    DcgmNvSwitchManager.h
    dcgm_nvswitch_structs.h
)

target_sources(nvswitch_objects
    PRIVATE
        ${SRCS}
)

add_library(dcgmmodulenvswitch SHARED)
define_dcgm_module(dcgmmodulenvswitch)
target_link_libraries(dcgmmodulenvswitch
    PRIVATE
        sdk_nscq_loader
        nvswitch_interface
        module_common_objects
        fmt::fmt
)
target_sources(dcgmmodulenvswitch
    PRIVATE
        ${SRCS}
)
update_lib_ver(dcgmmodulenvswitch)

add_subdirectory(tests)
