# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

add_executable(mock_dcgm_ops_tests)
target_sources(mock_dcgm_ops_tests
    PRIVATE
        MockDcgmOpsTests.cpp
)

target_link_libraries(mock_dcgm_ops_tests
    PRIVATE
        Catch2::Catch2WithMain
        common_interface
        mock_dcgm_ops
        dcgm_logging
        dcgm_common
        dcgm
        dl
        fmt::fmt
        rt)

target_link_options(mock_dcgm_ops_tests PRIVATE -rdynamic)

if (CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
    catch_discover_tests(mock_dcgm_ops_tests EXTRA_ARGS --colour-mode ansi)
endif()
