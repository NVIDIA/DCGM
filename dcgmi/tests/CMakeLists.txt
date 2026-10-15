# Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
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

if (NOT BUILD_TESTING)
    return()
endif()

add_executable(dcgmitests)
target_sources(dcgmitests
    PRIVATE
    CommandLineParserTests.cpp
    ConfigTests.cpp
    DcgmiCommonTests.cpp
    DcgmiProfileTests.cpp
    DcgmiSettingsTests.cpp
    DcgmiTestTests.cpp
    DeviceMonitorTests.cpp
    DiagTests.cpp
    EnvironmentTests.cpp
    FieldGroupTests.cpp
    GroupTests.cpp
    HealthTests.cpp
    IntrospectTests.cpp
    MnDiagTests.cpp
    ModuleTests.cpp
    NvcmTCLAPTests.cpp
    NvlinkTests.cpp
    OutputTests.cpp
    PolicyTests.cpp
    ProcessStatsTests.cpp
    QueryTests.cpp
    TopoTests.cpp
    VersionTests.cpp
)

target_link_libraries(dcgmitests
    PRIVATE
        dcgmi_objects_without_main
        nvvs_interface
        diag_interface
        dcgmmodulediag_common
        buildinfo_objects
        sdk_nvml_essentials_objects
        sdk_nvml_interface
        Catch2::Catch2WithMain
        test-utils
        dcgm
        dcgm_common
        dcgm_logging
        dcgm_mutex
        mock_dcgm_ops
        mock_common
        Threads::Threads
        jsoncpp_static
        fmt::fmt
        rt
        dl
)
if (${LINK_INJECTION_LIB} STREQUAL "true")
    target_link_libraries(dcgmitests PRIVATE nvml_injection)
endif()

if (CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
    catch_discover_tests(dcgmitests EXTRA_ARGS --colour-mode ansi)
endif()
