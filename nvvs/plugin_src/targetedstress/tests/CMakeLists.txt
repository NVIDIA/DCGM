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

add_executable(targetedstresstests)
target_sources(targetedstresstests
    PRIVATE
        TargetedStressPluginTests.cpp
        ../TargetedStress_wrapper.cpp
)

target_link_libraries(targetedstresstests
    PRIVATE
        nvvs_without_main_objects
        nvvs_plugins_targetedstress_interface
        Catch2::Catch2WithMain
        mock_cuda_ops
        mock_cuda_common
        mock_cublas_ops
        mock_common
        mock_dcgm_ops
        test-utils
)
target_link_libraries(targetedstresstests PUBLIC nvvs_interface)
target_link_libraries(targetedstresstests PRIVATE yaml-cpp::yaml-cpp jsoncpp_static rt dl)
target_link_libraries(targetedstresstests PRIVATE Threads::Threads)
target_link_libraries(targetedstresstests PRIVATE sdk_nvml_loader)
target_link_libraries(targetedstresstests PRIVATE sdk_nvml_essentials_objects)
target_link_libraries(targetedstresstests PUBLIC nvvs_plugins_common_objects)

target_link_options(targetedstresstests PRIVATE -rdynamic)

if (CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
    catch_discover_tests(targetedstresstests EXTRA_ARGS --colour-mode ansi)
endif()
