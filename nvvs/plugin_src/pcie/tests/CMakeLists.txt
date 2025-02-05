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

add_executable(pcietests)
target_sources(pcietests
    PRIVATE
        ../PcieMain.cpp
        ../Pcie.cpp
        ../Brokenp2p.cpp
        PciePluginTests.cpp
        Brokenp2pTests.cpp
)

target_include_directories(pcietests PRIVATE ${YAML_INCLUDE_DIR})
target_include_directories(pcietests PRIVATE ${JSONCPP_INCLUDE_DIR})
target_include_directories(pcietests PRIVATE ${CUDA12_INCLUDE_DIR})
target_include_directories(pcietests PRIVATE ${PROJECT_SOURCE_DIR}/cublas_proxy)
target_include_directories(pcietests PRIVATE ..) # include the plugin dir
target_include_directories(pcietests PRIVATE ../../include) # include the plugin_src/include dir

# NOTE: linking dcgmlib is not allowed for this library. Instead, make a mock implementation
#       for any libdcgm function that is needed.
#
# NOTE: linking cuda is not allowed for this library. We are using a CUDA stub library instead (testing_cuda_stubs)
#
target_link_libraries(pcietests
    PRIVATE
        nvvs_without_main_objects
        Catch2::Catch2WithMain
        testing_cuda_stubs
        testing_dcgm_cublas_stubs
        cuda_common_interface
)

target_link_libraries(pcietests PUBLIC nvvs_interface)
target_link_libraries(pcietests PRIVATE ${YAML_STATIC_LIBS} ${JSONCPP_STATIC_LIBS} rt dl)
target_link_libraries(pcietests PRIVATE ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(pcietests PRIVATE sdk_nvml_loader)
target_link_libraries(pcietests PRIVATE sdk_nvml_essentials_objects)
target_link_libraries(pcietests PUBLIC nvvs_plugins_common_objects)

target_link_options(pcietests PRIVATE -rdynamic)

if (CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
    catch_discover_tests(pcietests EXTRA_ARGS --colour-mode ansi)
endif()

