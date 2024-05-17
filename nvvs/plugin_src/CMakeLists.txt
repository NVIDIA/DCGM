# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

set(NVVS_PLUGINS_BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

macro(declare_nvvs_plugin plugin_name dirpath)
    add_library(nvvs_plugins_${plugin_name}_interface INTERFACE)
    target_include_directories(nvvs_plugins_${plugin_name}_interface INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/${dirpath})
    target_link_libraries(nvvs_plugins_${plugin_name}_interface INTERFACE nvvs_interface nvvs_plugins_interface)
    set(AUX_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/${dirpath})
    get_property(LINKER_SCRIPT GLOBAL PROPERTY NVVS_PLUGINS_LINKER_SCRIPT)

    function (nvvs_plugin_srcs lib_name)
        aux_source_directory(${AUX_SRC_DIR} LOCAL_SRCS)
        target_sources(${lib_name} PRIVATE ${LOCAL_SRCS})
        target_link_libraries(${lib_name} PUBLIC nvvs_plugins_${plugin_name}_interface)
        target_link_libraries(${lib_name} PRIVATE ${ARGN} dcgm_mutex dcgm_thread dcgm_logging)
        target_include_directories(${lib_name} PRIVATE ${AUX_SRC_DIR})
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
    target_link_libraries(${PLUGIN_NAME}_${CUDA_VER} PRIVATE pluginCudaCommon_${CUDA_VER})
    target_link_libraries(${PLUGIN_NAME}_${CUDA_VER} PRIVATE dcgm_logging)
    target_link_libraries(${PLUGIN_NAME}_${CUDA_VER} PRIVATE dcgm_common ${JSONCPP_STATIC_LIBS})
    set_target_properties(${PLUGIN_NAME}_${CUDA_VER} PROPERTIES LIBRARY_OUTPUT_NAME "${PLUGIN_NAME}")
    update_lib_ver(${PLUGIN_NAME}_${CUDA_VER})
    set_target_properties(${PLUGIN_NAME}_${CUDA_VER} PROPERTIES INSTALL_TO_DIR ${DCGM_NVVS_PLUGINS_INSTALL_DIR}/cuda${CUDA_VER})
    set(LOCAL_NVVS_PLUGINS "${LOCAL_NVVS_PLUGINS};${PLUGIN_NAME}_${CUDA_VER}" PARENT_SCOPE)
    if ((${CUDA_VER} EQUAL 10) OR (${CUDA_VER} EQUAL 9))
        target_compile_options(${PLUGIN_NAME}_${CUDA_VER} PRIVATE -Wno-volatile)
    endif()

    target_link_options(${PLUGIN_NAME}_${CUDA_VER} PRIVATE -Wl,--version-script,${NVVS_PLUGINS_BASE_DIR}/nvvs_plugin.linux_def)
endmacro()

LIST(APPEND SUPPORTED_CUDA_VERSIONS 12)
LIST(APPEND SUPPORTED_CUDA_VERSIONS 11)

LIST(SORT SUPPORTED_CUDA_VERSIONS)

LIST(APPEND NVVS_PLUGINS "")
LIST(APPEND BW_CHECKERS "")

add_subdirectory(test_common)
add_subdirectory(common)
add_subdirectory(cuda_common)
add_subdirectory(pcie)
add_subdirectory(targetedstress)
add_subdirectory(targetedpower)
add_subdirectory(contextcreate)
add_subdirectory(diagnostic)
add_subdirectory(memory)
#add_subdirectory(smstress)
add_subdirectory(software)
add_subdirectory(memtest)

foreach(plugin ${NVVS_PLUGINS})
    install(TARGETS ${plugin}
        LIBRARY DESTINATION $<TARGET_PROPERTY:${plugin},INSTALL_TO_DIR>
        PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
        COMPONENT DCGM)
endforeach()

# Install the BwChecker binary
foreach(bwchecker ${BW_CHECKERS})
    install(TARGETS ${bwchecker}
        RUNTIME DESTINATION $<TARGET_PROPERTY:${bwchecker},INSTALL_TO_DIR>
        PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
        COMPONENT DCGM)

    set_target_properties(${bwchecker} PROPERTIES INSTALL_RPATH "${DCGM_RPATH}:$ORIGIN/../../../../apps/${DCGM_TESTS_ARCH}")

    get_target_property(INSTALL_TO_DIR ${bwchecker} INSTALL_TO_DIR)
    string(
        REPLACE
            "${DCGM_NVVS_PLUGINS_INSTALL_DIR}"
            "${DCGM_TESTS_INSTALL_DIR}/apps/nvvs/plugins"
            INSTALL_TO
            ${INSTALL_TO_DIR}
    )
    install(TARGETS ${bwchecker}
        RUNTIME DESTINATION ${INSTALL_TO}
        PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
        COMPONENT Tests)
endforeach()

file(COPY ${DCGMPROFTESTER_PTX} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
foreach(cuda_ver IN LISTS SUPPORTED_CUDA_VERSIONS)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/DcgmProfTesterKernels.ptx DESTINATION ${DCGM_NVVS_PLUGINS_INSTALL_DIR}/cuda${cuda_ver} COMPONENT DCGM)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/DcgmProfTesterKernels.ptx DESTINATION ${DCGM_TESTS_INSTALL_DIR}/apps/nvvs/plugins/cuda${cuda_ver} COMPONENT Tests)
endforeach()

set(NVVS_PLUGINS ${NVVS_PLUGINS} PARENT_SCOPE)
set(NVVS_PLUGINS_BASE_DIR ${NVVS_PLUGINS_BASE_DIR} PARENT_SCOPE)

target_link_libraries(
    nvvs_plugins
    INTERFACE
        ${NVVS_PLUGINS}
)
