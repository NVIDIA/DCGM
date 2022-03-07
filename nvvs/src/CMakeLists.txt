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
set(CMAKE_CXX_VISIBILITY_PRESET default)
set(CMAKE_C_VISIBILITY_PRESET default)
set(CMAKE_VISIBILITY_INLINES_HIDDEN 0)

add_subdirectory(tests)

find_package(Yaml REQUIRED)
find_package(Jsoncpp REQUIRED)

file(READ ${CMAKE_CURRENT_BINARY_DIR}/../diag-skus.yaml FALLBACK_DIAG_YAML_CONTENTS_RAW)
string(REPLACE "\n"
       "\\n" FALLBACK_DIAG_YAML_CONTENTS
       "${FALLBACK_DIAG_YAML_CONTENTS_RAW}")

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/FallbackDiagConfig.cpp.in
    ${CMAKE_CURRENT_BINARY_DIR}/FallbackDiagConfig.cpp
    ESCAPE_QUOTES
    @ONLY)

add_library(nvvs_without_main_objects OBJECT)
target_link_libraries(nvvs_without_main_objects PUBLIC nvvs_interface dcgm dcgm_common dcgm_logging dcgm_mutex)
target_include_directories(nvvs_without_main_objects PRIVATE ${YAML_INCLUDE_DIR})
target_include_directories(nvvs_without_main_objects PRIVATE ${JSONCPP_INCLUDE_DIR})
target_sources(nvvs_without_main_objects
    PRIVATE
        ConfigFileParser_v2.cpp
        DcgmGdFieldGroup.cpp
        DcgmGroup.cpp
        DcgmHandle.cpp
        DcgmRecorder.cpp
        DcgmSystem.cpp
        DcgmValuesSinceHolder.cpp
        GoldenValueCalculator.cpp
        Gpu.cpp
        GpuSet.cpp
        JsonOutput.cpp
        NvidiaValidationSuite.cpp
        NvvsCommon.cpp
        NvvsDeviceList.cpp
        NvvsSystemChecker.cpp
        NvvsThread.cpp
        Output.cpp
        ParameterValidator.cpp
        ParsingUtility.cpp
        Plugin.cpp
        Test.cpp
        TestFramework.cpp
        TestParameters.cpp
        Whitelist.cpp
        PluginLib.cpp
        PluginCoreFunctionality.cpp
        CustomStatHolder.cpp
        ${CMAKE_CURRENT_BINARY_DIR}/FallbackDiagConfig.cpp
)

add_library(nvvs_main_objects OBJECT)
target_link_libraries(nvvs_main_objects PUBLIC nvvs_without_main_objects)
target_sources(
    nvvs_main_objects
    PRIVATE
    NvvsMain.cpp
)

add_executable(nvvs)
target_link_libraries(nvvs INTERFACE perfworks)
target_link_libraries(
    nvvs
    PRIVATE
        nvvs_without_main_objects
        nvvs_main_objects
)
target_link_libraries(nvvs INTERFACE nvvs_plugins)

find_package(Threads REQUIRED)

target_link_libraries(nvvs PRIVATE ${YAML_STATIC_LIBS} ${JSONCPP_STATIC_LIBS})
target_link_libraries(nvvs PRIVATE rt dl)
target_link_libraries(nvvs PRIVATE ${CMAKE_THREAD_LIBS_INIT})

target_link_options(nvvs PRIVATE -rdynamic)
