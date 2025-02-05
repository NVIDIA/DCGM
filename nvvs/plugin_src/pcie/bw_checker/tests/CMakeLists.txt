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

add_compile_definitions(__BW_CHECKER_TESTS__=1)

add_executable(bw_checkertests)
target_sources(bw_checkertests
    PRIVATE
        ../BwCheckerMain.cpp
        BwCheckerTests.cpp
)

target_link_libraries(bw_checkertests PRIVATE ${JSONCPP_STATIC_LIBS} rt dl Catch2::Catch2WithMain)

target_include_directories(bw_checkertests PRIVATE ${JSONCPP_INCLUDE_DIR})
target_include_directories(bw_checkertests PRIVATE ..) # include the bw_checker dir
target_link_libraries(bw_checkertests PRIVATE common_interface)
target_link_libraries(bw_checkertests PRIVATE dcgm_thread dcgm_common dcgm)
target_link_libraries(bw_checkertests PUBLIC dcgmi_interface)

target_link_options(bw_checkertests PRIVATE -rdynamic)

if (CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
    catch_discover_tests(bw_checkertests EXTRA_ARGS --colour-mode ansi)
endif()
