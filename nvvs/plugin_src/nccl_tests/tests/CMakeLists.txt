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

add_executable(ncclteststests)
target_sources(ncclteststests
    PRIVATE
        NcclTestsPluginTests.cpp
        NcclTestsWrapperTests.cpp
)

target_link_libraries(ncclteststests
    PRIVATE
        Catch2::Catch2WithMain
        test-utils
        nvvs_plugins_nccl_tests_interface
        nvvs_without_main_objects
        nvvs_plugins_common_objects
        mock_common
        mock_dcgm_ops
        rt
        yaml-cpp::yaml-cpp
)

if (CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
    catch_discover_tests(ncclteststests EXTRA_ARGS --colour-mode ansi)
endif()
