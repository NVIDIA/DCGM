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

if (NOT BUILD_TESTING)
    return()
endif()

add_executable(nvvscoretests)
target_sources(nvvscoretests
    PRIVATE
        NvvsTestsMain.cpp
        NvidiaValidationSuiteTests.cpp
        TestParametersTests.cpp
        DcgmRecorderTests.cpp
        DcgmDiagUnitTestCommon.cpp
        ConfigFileParser_v2Tests.cpp
        DcgmValuesSinceHolderTests.cpp
        AllowlistTests.cpp
        ParsingUtilityTests.cpp
        TestFrameworkTests.cpp
        PluginTests.cpp
        PluginTestTests.cpp
        PluginLibTests.cpp
        PluginLibTestTests.cpp
        PluginCoreFunctionalityTests.cpp
        CustomDataHolderTests.cpp
        SoftwarePluginFrameworkTests.cpp
        SoftwareTests.cpp
        CpuSetTests.cpp
        DcgmNvvsResponseWrapperTests.cpp
        IgnoreErrorCodesHelperTests.cpp
        NvvsDeviceListTests.cpp)

target_include_directories(nvvscoretests
    PRIVATE
        ${JSONCPP_INCLUDE_DIR}
        ${YAML_INCLUDE_DIR})

# NOTE: linking dcgmlib is not allowed for this library. Instead, make a mock implementation
#       for any libdcgm function that is needed.
target_link_libraries(nvvscoretests
    PRIVATE
        ${CMAKE_THREAD_LIBS_INIT}
        ${JSONCPP_STATIC_LIBS}
        ${YAML_STATIC_LIBS}
        Catch2::Catch2
        dcgm_logging
        dl
        fmt::fmt
        nvvs_without_main_objects
        rt
        sdk_nvml_loader
        serialize)

target_link_options(nvvscoretests PRIVATE -rdynamic)

add_dependencies(nvvscoretests testplugin)

add_library(testplugin SHARED)
set_target_properties(testplugin PROPERTIES
    CXX_VISIBILITY_PRESET default
    SOVERSION ${PROJECT_VERSION_MAJOR})

target_link_libraries(testplugin
    PUBLIC
        nvvs_interface
    PRIVATE
        fmt::fmt)

target_sources(testplugin PRIVATE fake_plugin.cpp)

if (CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
    catch_discover_tests(nvvscoretests EXTRA_ARGS --colour-mode ansi)
endif()
