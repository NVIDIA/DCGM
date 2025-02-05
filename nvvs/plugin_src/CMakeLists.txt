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

target_include_directories(nvvs_plugins_interface INTERFACE include)

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

    target_include_directories(${PLUGIN_NAME}_${CUDA_VER} PRIVATE
        "${CUDA${CUDA_VER}_INCLUDE_DIR}")

    target_link_libraries(${PLUGIN_NAME}_${CUDA_VER}
        PRIVATE
            ${CUDA${CUDA_VER}_STATIC_LIBS}
            ${JSONCPP_STATIC_LIBS}
            dcgm_common
            dcgm_cublas_proxy${CUDA_VER}
            dcgm_logging
            pluginCudaCommon_${CUDA_VER}
            sdk_nvml_essentials_objects
            sdk_nvml_loader
        PUBLIC
            ${CUDA${CUDA_VER}_LIBS}
            pluginCommon)

    set_target_properties(${PLUGIN_NAME}_${CUDA_VER} PROPERTIES
        LIBRARY_OUTPUT_NAME "${PLUGIN_NAME}"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")

    target_link_options(${PLUGIN_NAME}_${CUDA_VER} PRIVATE 
        -Wl,--version-script,${PROJECT_SOURCE_DIR}/nvvs/plugin_src/nvvs_plugin.linux_def)

    target_link_libraries(nvvs_plugins INTERFACE "${PLUGIN_NAME}_${CUDA_VER}")
endmacro()

set(SUPPORTED_CUDA_VERSIONS 11 12)

add_subdirectory(common)
add_subdirectory(contextcreate)
add_subdirectory(cuda_common)
add_subdirectory(diagnostic)
add_subdirectory(memory)
add_subdirectory(memtest)
add_subdirectory(nvbandwidth)
add_subdirectory(pcie)
add_subdirectory(targetedpower)
add_subdirectory(targetedstress)
add_subdirectory(test_common)

set(NVVS_PLUGINS_BASE_DIR ${NVVS_PLUGINS_BASE_DIR} PARENT_SCOPE)
