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

add_executable(healthtests)
target_sources(healthtests
    PRIVATE
        DcgmHealthWatchTests.cpp
        ../DcgmHealthWatch.cpp
        ../DcgmHealthResponse.cpp
        ../DcgmModuleHealth.cpp
        DcgmModuleHealthTests.cpp
)

target_link_libraries(healthtests
    PRIVATE
        dcgm_logging
        Catch2::Catch2WithMain
        fmt::fmt
        dcgm_interface
        module_common_interface
        dcgm_common
        health_interface
        diag_interface
        childprocess
        mock_file_handle
        module_common_objects
        dcgm
        dcgm_mutex
        mock_common
        modules_objects
)
target_link_libraries(healthtests PRIVATE yaml-cpp::yaml-cpp jsoncpp_static rt dl)
target_link_libraries(healthtests PRIVATE Threads::Threads)
target_link_libraries(healthtests PRIVATE sdk_nvml_loader)
target_link_options(healthtests PRIVATE -rdynamic)

if (CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
    catch_discover_tests(healthtests EXTRA_ARGS --colour-mode ansi)
endif()
