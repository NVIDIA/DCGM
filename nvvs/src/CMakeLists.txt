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

add_subdirectory(tests)

file(READ "${CMAKE_CURRENT_BINARY_DIR}/../diag-skus.yaml" FALLBACK_DIAG_YAML_CONTENTS_RAW)

string(REPLACE "\n"
       "\\n" FALLBACK_DIAG_YAML_CONTENTS
       "${FALLBACK_DIAG_YAML_CONTENTS_RAW}")

configure_file(FallbackDiagConfig.cpp.in FallbackDiagConfig.cpp ESCAPE_QUOTES @ONLY)
target_link_libraries(nvvs_without_main_objects
    PUBLIC
        dcgm
        dcgm_common
        dcgm_logging
        dcgm_mutex
        nvvs_interface
        serialize)

target_include_directories(nvvs_without_main_objects
    PRIVATE
        "${YAML_INCLUDE_DIR}"
        "${JSONCPP_INCLUDE_DIR}")

target_sources(nvvs_without_main_objects
    PRIVATE
        ConfigFileParser_v2.cpp
        CpuSet.cpp
        DcgmGdFieldGroup.cpp
        DcgmGroup.cpp
        DcgmHandle.cpp
        DcgmRecorder.cpp
        DcgmSystem.cpp
        DcgmValuesSinceHolder.cpp
        EarlyFailChecker.cpp
        EntitySet.cpp
        Gpu.cpp
        GpuSet.cpp
        IgnoreErrorCodesHelper.cpp
        NvidiaValidationSuite.cpp
        NvvsCommon.cpp
        NvvsDeviceList.cpp
        ParameterValidator.cpp
        ParsingUtility.cpp
        Plugin.cpp
        PluginTest.cpp
        Test.cpp
        TestFramework.cpp
        TestParameters.cpp
        Allowlist.cpp
        PluginLib.cpp
        PluginLibTest.cpp
        PluginCoreFunctionality.cpp
        CustomStatHolder.cpp
        SoftwarePluginFramework.cpp
        Software.cpp
        FdChannelClient.cpp
        DcgmNvvsResponseWrapper.cpp
        ResultHelpers.cpp
        "${CMAKE_CURRENT_BINARY_DIR}/FallbackDiagConfig.cpp")

target_link_libraries(nvvs_main_objects PUBLIC nvvs_without_main_objects)
target_sources(nvvs_main_objects PRIVATE NvvsMain.cpp)

target_link_libraries(nvvs
    INTERFACE
        nvvs_plugins
        perfworks
    PRIVATE
        ${CMAKE_THREAD_LIBS_INIT}
        ${JSONCPP_STATIC_LIBS}
        ${YAML_STATIC_LIBS}
        dl
        nvvs_main_objects
        nvvs_without_main_objects
        rt
        sdk_nvml_loader)

target_link_options(nvvs
    PRIVATE
        -rdynamic
        -Wl,--version-script,${CMAKE_CURRENT_SOURCE_DIR}/../nvvs.linux_def)
