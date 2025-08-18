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

add_executable(nvswitchtests)
target_sources(nvswitchtests
    PRIVATE
        DcgmNscqManagerTests.cpp
        DcgmNvsdmManagerTests.cpp
        DcgmModuleNvSwitchTests.cpp
)
    
target_link_libraries(nvswitchtests
    PRIVATE
        nvswitch_interface
        sdk_nvml_interface
        sdk_nscq_interface

        nvswitch_objects
        sdk_nscq_loader
        common_watch_objects
        module_common_objects
        modules_objects
        dcgm_common
        dcgm_logging
        dcgm_mutex
        dcgm
        sdk_nvml_essentials_objects
        sdk_nvml_loader
        Catch2::Catch2WithMain
        ${CMAKE_THREAD_LIBS_INIT}
        fmt::fmt
        rt
        dl
)

if (CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
    catch_discover_tests(nvswitchtests EXTRA_ARGS --colour-mode ansi)
endif()
