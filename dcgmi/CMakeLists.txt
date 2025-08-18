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
add_subdirectory(testing)

target_include_directories(dcgmi_interface INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(dcgmi_interface INTERFACE dcgm_interface)

target_link_libraries(dcgmi_objects_without_main PUBLIC
    common_interface
    dcgm_common
    dcgm_logging
    dcgm_thread_interface
    dcgmi_interface
    dcgmmodulediag_common
    diag_interface
    dcgmmodulemndiag_common
    mndiag_interface
    ${JSONCPP_STATIC_LIBS}
    nvmli_interface)

target_sources(dcgmi_objects_without_main PRIVATE
    Command.cpp
    CommandLineParser.cpp
    CommandOutputController.cpp
    Config.cpp
    dcgmi_common.cpp
    DcgmiOutput.cpp
    DcgmiProfile.cpp
    DcgmiSettings.cpp
    DcgmiTest.cpp
    DeviceMonitor.cpp
    Diag.cpp
    FieldGroup.cpp
    Group.cpp
    Health.cpp
    Introspect.cpp
    MnDiag.cpp
    Module.cpp
    Nvlink.cpp
    Policy.cpp
    ProcessStats.cpp
    Query.cpp
    Topo.cpp
    Version.cpp)

target_link_libraries(dcgmi_objects PUBLIC dcgmi_objects_without_main)
target_sources(dcgmi_objects PRIVATE main_dcgmi.cpp)

target_link_libraries(dcgmi
    PRIVATE
        buildinfo_objects
        common_interface
        dcgm
        dcgm_interface
        dcgm_mutex
        dcgm_thread
        dcgmi_objects
        fmt::fmt
        nvvs_interface)

target_link_options(dcgmi PRIVATE -Wl,--version-script,${CMAKE_CURRENT_SOURCE_DIR}/dcgmi.linux_def)

add_subdirectory(tests)
