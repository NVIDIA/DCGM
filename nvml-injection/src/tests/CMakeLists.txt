#
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
#

if (NOT BUILD_TESTING)
    return()
endif()

add_definitions(-DUNIT_TEST)

add_executable(nvmlicoretests)
target_sources(nvmlicoretests
    PRIVATE
        InjectedNvmlTests.cpp
        InjectionArgumentCoverageTests.cpp
        InjectionArgumentGeneratedArrayTests.cpp
        InjectionArgumentPrimitiveOwnershipTests.cpp
        InjectionArgumentGeneratedStructTests.cpp
        NvmlInjectionLogging.cpp
        NvmlLoggerDependency.cpp
        NvmlInjectionUtilTests.cpp
        NvmlReturnDeserializerCoverageTests.cpp
        NvmlGeneratedStubsCoverageTests.cpp
        ../CompoundValue.cpp
        ../InjectedNvml.cpp
        ../NvmlLogging.cpp
        ../InjectionArgument.cpp
        ../InjectionKeys.cpp
        ../NvmlFuncReturn.cpp
        ../NvmlReturnDeserializer.cpp
        ../NvmlInjectionUtil.cpp
        ../nvml_generated_stubs.cpp
        ../nvml_stubs.cpp
        ../PassThruNvml.cpp
        ../UnitTestHelpers.cpp
)

target_include_directories(nvmlicoretests
    PRIVATE
        ..
        ${UNIT_TEST_HELPERS_HEADER_DIR} # for UnitTestHelpers.cpp to be able to include generated header (WithInjectionSku.h).
)

target_link_libraries(nvmlicoretests
    PRIVATE
        Catch2::Catch2WithMain
        test-utils
        yaml-cpp::yaml-cpp
        nvmli_interface
        sdk_nvml_interface
        dcgm_common
        dcgm
        common_interface
        mock_common
        rt
        dl
        sdk_nvml_essentials_objects
        fmt::fmt
)

if (CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
    catch_discover_tests(nvmlicoretests
        PROPERTIES SKIP_REGULAR_EXPRESSION "[1-9][0-9]* skipped"
        EXTRA_ARGS --colour-mode ansi)
endif()
