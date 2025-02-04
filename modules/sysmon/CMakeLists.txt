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

target_include_directories(sysmon_interface INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(sysmon_interface
    INTERFACE
        dcgm_interface
        common_interface
        dcgm_thread_interface
        modules_interface
        module_common_interface
        transport_interface
)

target_link_libraries(sysmon_objects
    PRIVATE
        sysmon_interface
        ${JSONCPP_STATIC_LIBS}
)

set(SRCS 
    DcgmModuleSysmon.cpp
    DcgmCpuManager.cpp
    DcgmCpuTopology.cpp
    DcgmSystemMonitor.cpp
)

target_sources(sysmon_objects
    PRIVATE
        ${SRCS}
)

define_dcgm_module(dcgmmodulesysmon)
target_link_libraries(dcgmmodulesysmon
    PRIVATE
        sysmon_interface
        module_common_objects
        dcgm_entity_types
        dcgm_thread
        fmt::fmt
        ${JSONCPP_STATIC_LIBS}
)
target_sources(dcgmmodulesysmon
    PRIVATE
        ${SRCS}
)

add_subdirectory(tests)
