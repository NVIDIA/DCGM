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

add_executable(dcgmitests)
target_sources(dcgmitests
    PRIVATE
    DiagTests.cpp
    HealthTests.cpp
    OutputTests.cpp
    MigTestsHelper.cpp
    QueryTests.cpp
    CommandLineParserTests.cpp
)

# NOTE: linking dcgmlib is not allowed for this library. Instead, make a mock implementation
#       for any libdcgm function that is needed.
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
        dcgm
        dcgm_common
        dcgm_logging
        dcgm_mutex
        ${CMAKE_THREAD_LIBS_INIT}
        ${JSONCPP_STATIC_LIBS}
        fmt::fmt
        rt
        dl
)
if (${LINK_INJECTION_LIB} STREQUAL "true")
    target_link_libraries(dcgmitests PRIVATE nvml_injection)
endif()
#target_include_directories(dcgmitests PRIVATE ${JSONCPP_INCLUDE_DIR})

if (CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
    catch_discover_tests(dcgmitests EXTRA_ARGS --colour-mode ansi)
endif()
