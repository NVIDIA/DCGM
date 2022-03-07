# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
set(CTEST_USE_LAUNCHERS 1)

find_package(Catch2 REQUIRED)
find_package(Yaml REQUIRED)
find_package(Jsoncpp REQUIRED)
find_package(Cuda REQUIRED)

include(CTest)
include(Catch)

if (BUILD_TESTING)
    add_executable(diagnostictests)
    target_sources(diagnostictests
        PRIVATE
            DiagnosticPluginTests.cpp
            ../DiagnosticPlugin.cpp
            DiagnosticTestsMain.cpp
    )

    target_include_directories(diagnostictests PRIVATE ${YAML_INCLUDE_DIR})
    target_include_directories(diagnostictests PRIVATE ${JSONCPP_INCLUDE_DIR})
    target_include_directories(diagnostictests PRIVATE ${CUDA11_INCLUDE_DIR})
    target_include_directories(diagnostictests PRIVATE ${PROJECT_SOURCE_DIR}/cublas_proxy)
    target_include_directories(diagnostictests PRIVATE ..) # include the plugin dir
    target_include_directories(diagnostictests PRIVATE ../../include) # include the plugin_src/include dir

    # NOTE: linking dcgmlib is not allowed for this library. Instead, make a mock implementation
    #       for any libdcgm function that is needed.    
    target_link_libraries(diagnostictests
        PRIVATE
            nvvs_without_main_objects
            Catch2::Catch2
    )
    target_link_libraries(diagnostictests PUBLIC nvvs_interface)
    target_link_libraries(diagnostictests PRIVATE ${YAML_STATIC_LIBS} ${JSONCPP_STATIC_LIBS} rt dl)
    target_link_libraries(diagnostictests PRIVATE ${CMAKE_THREAD_LIBS_INIT})
    target_link_libraries(diagnostictests PRIVATE dcgm_cublas_proxy11)
    target_link_libraries(diagnostictests PRIVATE ${CUDA11_STATIC_LIBS})
    target_link_libraries(diagnostictests PRIVATE sdk_nvml_essentials_objects)
    target_link_libraries(diagnostictests PUBLIC ${CUDA11_LIBS})
    target_link_libraries(diagnostictests PUBLIC pluginCommon_11)
    
    target_link_options(diagnostictests PRIVATE -rdynamic)
    if (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "x86_64")
        catch_discover_tests(diagnostictests EXTRA_ARGS --use-colour yes)
    endif()
endif()
