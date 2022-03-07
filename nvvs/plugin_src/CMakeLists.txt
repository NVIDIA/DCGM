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

find_package(Cuda REQUIRED)

add_library(nvvs_plugins INTERFACE)

add_library(nvvs_plugins_interface INTERFACE)
target_include_directories(nvvs_plugins_interface INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

macro(declare_nvvs_plugin plugin_name)
    add_library(nvvs_plugins_${plugin_name}_interface INTERFACE)
    target_include_directories(nvvs_plugins_${plugin_name}_interface INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(nvvs_plugins_${plugin_name}_interface INTERFACE nvvs_interface nvvs_plugins_interface)
    set(AUX_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR})
    get_property(LINKER_SCRIPT GLOBAL PROPERTY NVVS_PLUGINS_LINKER_SCRIPT)

    function (nvvs_plugin_srcs lib_name)
        aux_source_directory(${AUX_SRC_DIR} LOCAL_SRCS)
        target_sources(${lib_name} PRIVATE ${LOCAL_SRCS})
        target_link_libraries(${lib_name} PRIVATE nvvs_plugins_${plugin_name}_interface)
        target_link_libraries(${lib_name} PRIVATE ${ARGN} dcgm_mutex)
    endfunction()

endmacro()

macro(define_plugin PLUGIN_NAME CUDA_VER)
    add_library(${PLUGIN_NAME}_${CUDA_VER} SHARED)
    nvvs_plugin_srcs(${PLUGIN_NAME}_${CUDA_VER})
    target_include_directories(${PLUGIN_NAME}_${CUDA_VER} PRIVATE ${CUDA${CUDA_VER}_INCLUDE_DIR})
    target_link_libraries(${PLUGIN_NAME}_${CUDA_VER} PRIVATE dcgm_cublas_proxy${CUDA_VER})
    target_link_libraries(${PLUGIN_NAME}_${CUDA_VER} PRIVATE ${CUDA${CUDA_VER}_STATIC_LIBS})
    target_link_libraries(${PLUGIN_NAME}_${CUDA_VER} PRIVATE sdk_nvml_essentials_objects)
    target_link_libraries(${PLUGIN_NAME}_${CUDA_VER} PUBLIC ${CUDA${CUDA_VER}_LIBS})
    target_link_libraries(${PLUGIN_NAME}_${CUDA_VER} PUBLIC pluginCommon_${CUDA_VER})
    set_target_properties(${PLUGIN_NAME}_${CUDA_VER} PROPERTIES LIBRARY_OUTPUT_NAME "${PLUGIN_NAME}")
    update_lib_ver(${PLUGIN_NAME}_${CUDA_VER})
    set_target_properties(${PLUGIN_NAME}_${CUDA_VER} PROPERTIES INSTALL_TO_DIR ${DCGM_NVVS_PLUGINS_INSTALL_DIR}/cuda${CUDA_VER})
    set(LOCAL_NVVS_PLUGINS "${LOCAL_NVVS_PLUGINS};${PLUGIN_NAME}_${CUDA_VER}" PARENT_SCOPE)
endmacro()

LIST(APPEND SUPPORTED_CUDA_VERSIONS 11)
IF (NOT ${CMAKE_SYSTEM_PROCESSOR} STREQUAL "aarch64")
    LIST(APPEND SUPPORTED_CUDA_VERSIONS 9 10)
ENDIF()

LIST(SORT SUPPORTED_CUDA_VERSIONS)

LIST(APPEND NVVS_PLUGINS "")

add_subdirectory(common)
add_subdirectory(pcie)
add_subdirectory(targetedstress)
add_subdirectory(targetedpower)
add_subdirectory(contextcreate)
add_subdirectory(diagnostic)
add_subdirectory(memory)
add_subdirectory(smstress)
add_subdirectory(software)

foreach(plugin ${NVVS_PLUGINS})
    install(TARGETS ${plugin}
        LIBRARY DESTINATION $<TARGET_PROPERTY:${plugin},INSTALL_TO_DIR>
        PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
        COMPONENT DCGM)
endforeach()

set(NVVS_PLUGINS ${NVVS_PLUGINS} PARENT_SCOPE)

target_link_libraries(
    nvvs_plugins
    INTERFACE
        ${NVVS_PLUGINS}
)
