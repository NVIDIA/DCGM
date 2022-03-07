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
set(CTEST_USE_LAUNCHERS 1)

find_package(Catch2 REQUIRED)
find_package(Yaml REQUIRED)
find_package(Jsoncpp REQUIRED)
find_package(fmt REQUIRED)

include(CTest)
include(Catch)

if (BUILD_TESTING)

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
            WhitelistTests.cpp
            ParsingUtilityTests.cpp
            TestFrameworkTests.cpp
            GoldenValuesCalculatorTests.cpp
            PluginLibTests.cpp
            PluginCoreFunctionalityTests.cpp
            CustomDataHolderTests.cpp
    )

    target_include_directories(nvvscoretests PRIVATE ${YAML_INCLUDE_DIR})
    target_include_directories(nvvscoretests PRIVATE ${JSONCPP_INCLUDE_DIR})

    # NOTE: linking dcgmlib is not allowed for this library. Instead, make a mock implementation
    #       for any libdcgm function that is needed.    
    target_link_libraries(nvvscoretests
        PRIVATE
            nvvs_without_main_objects
            Catch2::Catch2
            fmt::fmt
    )
    target_link_libraries(nvvscoretests PRIVATE ${YAML_STATIC_LIBS} ${JSONCPP_STATIC_LIBS} rt dl)
    target_link_libraries(nvvscoretests PRIVATE ${CMAKE_THREAD_LIBS_INIT})
    target_link_options(nvvscoretests PRIVATE -rdynamic)

    add_dependencies(nvvscoretests testplugin)

    add_library(testplugin SHARED)

    target_link_libraries(testplugin PUBLIC nvvs_interface)

    target_sources(
        testplugin
        PRIVATE
        fake_plugin.cpp
    )

    if (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "x86_64")
        catch_discover_tests(nvvscoretests EXTRA_ARGS --use-colour yes)
    endif()
endif()
