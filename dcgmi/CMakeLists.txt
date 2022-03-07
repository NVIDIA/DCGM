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
add_subdirectory(testing)

find_package(Jsoncpp REQUIRED)

add_library(dcgmi_interface INTERFACE)
target_include_directories(dcgmi_interface INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(dcgmi_interface INTERFACE dcgm_interface)

add_library(dcgmi_objects STATIC)
target_link_libraries(dcgmi_objects
    PUBLIC
        dcgmi_interface
        common_interface
        diag_interface
)
target_sources(dcgmi_objects
    PRIVATE
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
        main_dcgmi.cpp
        MigIdParser.cpp
        Module.cpp
        Nvlink.cpp
        Policy.cpp
        ProcessStats.cpp
        Query.cpp
        Topo.cpp
        Version.cpp
)

add_executable(dcgmi)
target_link_libraries(dcgmi
    PRIVATE
        dcgm_interface
        common_interface
        diag_interface

        dcgmi_objects

        dcgmmodulediag_common
        buildinfo_objects
        dcgm
        dcgm_logging
        dcgm_mutex
        dcgm_common
        nvvs_interface
        diag_interface
        ${JSONCPP_STATIC_LIBS}
)

target_link_options(dcgmi PRIVATE -Wl,--version-script,${CMAKE_CURRENT_SOURCE_DIR}/dcgmi.linux_def)

add_subdirectory(tests)
