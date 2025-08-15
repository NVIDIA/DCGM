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

add_executable(diagtests)
target_sources(diagtests
    PRIVATE
        DcgmDiagResponseWrapperTests.cpp
        ../DcgmDiagResponseWrapper.cpp
        DcgmDiagManagerTests.cpp
        ../DcgmDiagManager.cpp
)

target_include_directories(diagtests PRIVATE ${YAML_INCLUDE_DIR})
target_include_directories(diagtests PRIVATE ${JSONCPP_INCLUDE_DIR})
target_include_directories(diagtests PRIVATE ..)

# NOTE: linking dcgmlib is not allowed for this library. Instead, make a mock implementation
#       for any libdcgm function that is needed.
target_link_libraries(diagtests
    PRIVATE
        dcgm_logging
        Catch2::Catch2WithMain
        fmt::fmt
        dcgm_interface
        module_common_interface
        dcgm_common
        diag_interface
        childprocess
        module_common_objects
        dcgm
        dcgm_mutex
        mock_interface
)
target_link_libraries(diagtests PRIVATE ${YAML_STATIC_LIBS} ${JSONCPP_STATIC_LIBS} rt dl)
target_link_libraries(diagtests PRIVATE ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(diagtests PRIVATE sdk_nvml_loader)
target_link_options(diagtests PRIVATE -rdynamic)

if (CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
    catch_discover_tests(diagtests EXTRA_ARGS --colour-mode ansi)
endif()
