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

target_include_directories(nvswitch_interface INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(nvswitch_interface
    INTERFACE
        dcgm_interface
        common_interface
        dcgm_thread_interface
        modules_interface
        module_common_interface
        sdk_nscq_interface
        sdk_nvsdm_interface
        sdk_nvsdm_loader
        transport_interface
)

target_link_libraries(nvswitch_objects
    PRIVATE
        nvswitch_interface
        ${YAML_STATIC_LIBS}
)

set(SRCS 
    DcgmModuleNvSwitch.cpp
    DcgmModuleNvSwitch.h
    DcgmNvSwitchManagerBase.cpp
    DcgmNscqManager.cpp
    DcgmNvsdmManager.cpp
    NvSwitchData.cpp
    NvSwitchData.h
    UpdateFunctions.h
    FieldIds.h
    FieldDefinitions.h
    FieldDefinitions.cpp
    dcgm_nvswitch_structs.h
    NvsdmLib.cpp
    NvsdmLib.h
)

target_sources(nvswitch_objects
    PRIVATE
        ${SRCS}
)
target_include_directories(nvswitch_objects PRIVATE ${YAML_INCLUDE_DIR})

define_dcgm_module(dcgmmodulenvswitch)
target_link_libraries(dcgmmodulenvswitch
    PRIVATE
        sdk_nscq_loader
        sdk_nvsdm_loader
        nvswitch_interface
        module_common_objects
        dcgm_thread
        fmt::fmt
        ${YAML_STATIC_LIBS}
)
target_sources(dcgmmodulenvswitch
    PRIVATE
        ${SRCS}
)

add_subdirectory(tests)
