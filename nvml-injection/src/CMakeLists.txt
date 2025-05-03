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

set_property(
    TARGET nvml_injection nvmli_public
    APPEND PROPERTY COMPILE_DEFINITIONS NVML_NO_UNVERSIONED_FUNC_DEFS)

add_compile_definitions(NVML_NO_UNVERSIONED_FUNC_DEFS)

target_link_libraries(
    nvmli_public
    PRIVATE
        nvmli_interface
        sdk_nvml_interface
)

target_sources(nvmli_public
    PRIVATE
        InjectionArgument.cpp
        InjectionKeys.cpp
        NvmlInjectionUtil.cpp
        nvml_generated_stubs.cpp
        nvml_stubs.cpp
)

set_target_properties(nvml_injection PROPERTIES LINKER_LANGUAGE CXX)

target_sources(
    nvml_injection
    PRIVATE
        CompoundValue.cpp
        InjectedNvml.cpp
        NvmlFuncReturn.cpp
        NvmlReturnDeserializer.cpp
        PassThruNvml.cpp
)

target_link_libraries(nvml_injection PUBLIC nvmli_interface)
target_link_libraries(nvml_injection PRIVATE nvmli_public sdk_nvml_interface)
target_link_libraries(nvml_injection PRIVATE ${YAML_STATIC_LIBS})
target_link_libraries(nvml_injection PRIVATE -Wl,--version-script,${CMAKE_CURRENT_SOURCE_DIR}/nvml-injection.linux_defs)
target_link_libraries(nvml_injection PRIVATE fmt::fmt)
target_include_directories(nvml_injection PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(nvml_injection PRIVATE ${YAML_INCLUDE_DIR})

add_subdirectory(tests)
