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
add_subdirectory(testing)

find_package(Jsoncpp REQUIRED)

add_library(dcgmi_interface INTERFACE)
target_include_directories(dcgmi_interface INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(dcgmi_interface INTERFACE dcgm_interface)

add_library(dcgmi_objects OBJECT)
target_link_libraries(
    dcgmi_objects
    PUBLIC
        dcgmi_interface
        nvvs_interface
        diag_interface
        common_objects
        buildinfo_objects
)
target_sources(
    dcgmi_objects
    PRIVATE
        Command.cpp
        Command.h
        CommandLineParser.cpp
        CommandLineParser.h
        CommandOutputController.cpp
        CommandOutputController.h
        Config.cpp
        Config.h
        dcgmi_common.cpp
        dcgmi_common.h
        DcgmiOutput.cpp
        DcgmiOutput.h
        DcgmiProfile.cpp
        DcgmiProfile.h
        DcgmiSettings.cpp
        DcgmiSettings.h
        DcgmiTest.cpp
        DcgmiTest.h
        DeviceMonitor.cpp
        DeviceMonitor.h
        Diag.cpp
        Diag.h
        FieldGroup.cpp
        FieldGroup.h
        Group.cpp
        Group.h
        Health.cpp
        Health.h
        Introspect.cpp
        Introspect.h
        MigIdParser.cpp
        Module.cpp
        Module.h
        NvcmTCLAP.h
        Nvlink.cpp
        Nvlink.h
        Policy.cpp
        Policy.h
        ProcessStats.cpp
        ProcessStats.h
        Query.cpp
        Query.h
        Topo.cpp
        Topo.h
        Version.cpp
        Version.h
)

add_executable(dcgmi)
target_link_libraries(
    dcgmi
    PRIVATE
        dcgmi_interface
        nvvs_interface
        diag_interface
        dcgmi_objects
        dcgm
        dcgmmodulediag
        ${JSONCPP_STATIC_LIBS}
)
target_sources(dcgmi PRIVATE main_dcgmi.cpp)

add_subdirectory(tests)
