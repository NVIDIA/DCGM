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

add_executable(bw_checkertests)
target_sources(bw_checkertests
    PRIVATE
        BwCheckerTests.cpp
)

target_link_options(bw_checkertests PRIVATE -rdynamic)

target_link_libraries(bw_checkertests
    PRIVATE
        jsoncpp_static
        Catch2::Catch2WithMain
        common_interface
        dcgm
        dcgm_common
        dcgm_thread
        dcgmi_interface
        dl
        pcie_bw_checker
        pcie_bw_checker_interface
        rt)

if (CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
    catch_discover_tests(bw_checkertests EXTRA_ARGS --colour-mode ansi)
endif()
