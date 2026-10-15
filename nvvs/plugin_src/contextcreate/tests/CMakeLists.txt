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

add_executable(contextcreatetests)
target_sources(contextcreatetests
    PRIVATE
        ContextCreateTests.cpp
        ../ContextCreate.cpp
        ../ContextCreatePlugin.cpp
)

target_link_libraries(contextcreatetests
    PRIVATE
        nvvs_without_main_objects
        nvvs_plugins_contextcreate_interface
        Catch2::Catch2WithMain
        mock_cuda_ops
        mock_cuda_common
        common_interface
        mock_common
        mock_dcgm_ops
        test-utils
)
target_link_libraries(contextcreatetests PUBLIC nvvs_interface)
target_link_libraries(contextcreatetests PRIVATE yaml-cpp::yaml-cpp jsoncpp_static rt dl)
target_link_libraries(contextcreatetests PRIVATE Threads::Threads)
target_link_libraries(contextcreatetests PRIVATE sdk_nvml_loader)
target_link_libraries(contextcreatetests PRIVATE sdk_nvml_essentials_objects fmt::fmt)
target_link_libraries(contextcreatetests PUBLIC nvvs_plugins_common_objects)

target_link_options(contextcreatetests PRIVATE -rdynamic)
if (CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
    catch_discover_tests(contextcreatetests EXTRA_ARGS --colour-mode ansi)
endif()
