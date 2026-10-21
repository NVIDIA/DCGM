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

target_include_directories(nvvs_plugins_interface INTERFACE include)
target_link_libraries(nvvs_plugins_interface INTERFACE jsoncpp_static)

macro(declare_nvvs_plugin plugin_name dirpath)
    target_include_directories(nvvs_plugins_${plugin_name}_interface INTERFACE ${dirpath})
    target_link_libraries(nvvs_plugins_${plugin_name}_interface
        INTERFACE
            nvvs_interface
            nvvs_plugins_interface)

    set(AUX_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/${dirpath}")

    function (nvvs_plugin_srcs lib_name)
        target_link_libraries(${lib_name}
            PUBLIC
                nvvs_plugins_${plugin_name}_interface
            PRIVATE
                dcgm
                dcgm_mutex
                dcgm_thread
                dcgm_logging
                ${ARGN})

        target_include_directories(${lib_name} PRIVATE ${AUX_SRC_DIR})
        aux_source_directory("${AUX_SRC_DIR}" LOCAL_SRCS)
        target_sources(${lib_name} PRIVATE ${LOCAL_SRCS})
    endfunction()
endmacro()

macro(define_plugin PLUGIN_NAME CUDA_VER)
    cmake_parse_arguments(DEF_PLUGIN PRIVATE "" "" ${ARGN})

    nvvs_plugin_srcs(${PLUGIN_NAME}_${CUDA_VER})

    target_link_libraries(${PLUGIN_NAME}_${CUDA_VER}
        PRIVATE
            jsoncpp_static
            dcgm_common
            cuda_ops${CUDA_VER}
            dcgm_logging
            pluginCudaCommon_${CUDA_VER}
            sdk_nvml_essentials_objects
            sdk_nvml_loader
        PUBLIC
            pluginCommon)

    set_target_properties(${PLUGIN_NAME}_${CUDA_VER} PROPERTIES
        LIBRARY_OUTPUT_NAME "${PLUGIN_NAME}"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")

    nvvs_configure_plugin_version_script(${PLUGIN_NAME}_${CUDA_VER} ${ARGN})

    target_link_libraries(nvvs_plugins INTERFACE "${PLUGIN_NAME}_${CUDA_VER}")
endmacro()

# Creates the linker script for the plugin. ARGN consists of
# - (optional) DO_NOT_EXPORT <unexported symbol> ...: List of symbols to remove from the full list of exported symbols
function(nvvs_configure_plugin_version_script PLUGIN_NAME #[[ARGN]])
    cmake_parse_arguments(ARG "" "" "DO_NOT_EXPORT" ${ARGN})

    # Handle optionality
    if (NOT ARG_DO_NOT_EXPORT)
        set(ARG_DO_NOT_EXPORT "")
    endif()

    # Remove unexported functions
    set(EXPORTED_FUNCTIONS
        GetPluginInterfaceVersion
        GetPluginInfo
        InitializePlugin
        RunTest
        RetrieveCustomStats
        RetrieveResults
        ShutdownPlugin)
    foreach(UNEXPORTED_FUNCTION IN LISTS ARG_DO_NOT_EXPORT)
        list(REMOVE_ITEM EXPORTED_FUNCTIONS "${UNEXPORTED_FUNCTION}")
    endforeach()

    # Configure linker script and target
    set(EXPORTED_FUNCTIONS_FOR_LINKER_SCRIPT "")
    foreach(EXPORTED_FUNCTION IN LISTS EXPORTED_FUNCTIONS)
        string(APPEND EXPORTED_FUNCTIONS_FOR_LINKER_SCRIPT "        ${EXPORTED_FUNCTION};\n")
    endforeach()
    set(CONFIGURED_LINKER_SCRIPT "${CMAKE_CURRENT_BINARY_DIR}/${PLUGIN_NAME}.linux_def")
    configure_file(
        "${PROJECT_SOURCE_DIR}/nvvs/plugin_src/nvvs_plugin.linux_def.in"
        "${CONFIGURED_LINKER_SCRIPT}"
        @ONLY)
    target_link_options(${PLUGIN_NAME} PRIVATE
        "-Wl,--version-script,${CONFIGURED_LINKER_SCRIPT}")
endfunction()

set(SUPPORTED_CUDA_VERSIONS 11 12 13)

add_subdirectory(common)
add_subdirectory(contextcreate)
add_subdirectory(cuda_common)
add_subdirectory(diagnostic)
add_subdirectory(memory)
add_subdirectory(memtest)
add_subdirectory(nccl_tests)
add_subdirectory(nvbandwidth)
add_subdirectory(pcie)
add_subdirectory(cpu_sdc)
add_subdirectory(targetedpower)
add_subdirectory(targetedstress)

set(NVVS_PLUGINS_BASE_DIR ${NVVS_PLUGINS_BASE_DIR} PARENT_SCOPE)
