#
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
#

if (NOT BUILD_TESTING)
    return()
endif()

add_executable(nvmlicoretests)
target_sources(nvmlicoretests
    PRIVATE
        InjectedNvmlTests.cpp
        NvmlInjectionUtilTests.cpp
        ../CompoundValue.cpp
        ../InjectedNvml.cpp
        ../InjectionArgument.cpp
        ../InjectionKeys.cpp
        ../NvmlFuncReturn.cpp
        ../NvmlReturnDeserializer.cpp
        ../NvmlInjectionUtil.cpp
        ../nvml_generated_stubs.cpp
        ../nvml_stubs.cpp
        ../PassThruNvml.cpp
)

target_include_directories(nvmlicoretests
    PRIVATE
        ..
        ${YAML_INCLUDE_DIR}
)

target_link_libraries(nvmlicoretests
    PRIVATE
        Catch2::Catch2WithMain
        ${YAML_STATIC_LIBS}
        nvmli_interface
        sdk_nvml_interface
        rt
        dl
        sdk_nvml_essentials_objects
        fmt::fmt
)

if (CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
    catch_discover_tests(nvmlicoretests EXTRA_ARGS --colour-mode ansi)
endif()
