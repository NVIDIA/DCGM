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
declare_nvvs_plugin(common)

set(LOCAL_NVVS_PLUGINS "")

macro(common_plugin CUDA_VER)
    add_library(pluginCommon_${CUDA_VER} SHARED)
    nvvs_plugin_srcs(pluginCommon_${CUDA_VER})
    target_include_directories(pluginCommon_${CUDA_VER} PRIVATE ${CUDA${CUDA_VER}_INCLUDE_DIR})
    target_link_libraries(pluginCommon_${CUDA_VER} PRIVATE ${CUDA${CUDA_VER}_STATIC_LIBS})
    target_link_libraries(pluginCommon_${CUDA_VER} PUBLIC ${CUDA${CUDA_VER}_LIBS})
    set_target_properties(pluginCommon_${CUDA_VER} PROPERTIES LIBRARY_OUTPUT_NAME "pluginCommon")
    update_lib_ver(pluginCommon_${CUDA_VER})
    set_target_properties(pluginCommon_${CUDA_VER} PROPERTIES INSTALL_TO_DIR ${DCGM_NVVS_PLUGINS_INSTALL_DIR}/cuda${CUDA_VER})
    set(LOCAL_NVVS_PLUGINS "${LOCAL_NVVS_PLUGINS};pluginCommon_${CUDA_VER}" PARENT_SCOPE)
endmacro()

foreach(CUDA_VER IN LISTS SUPPORTED_CUDA_VERSIONS)
    add_subdirectory(Cuda${CUDA_VER})
endforeach()

set(NVVS_PLUGINS "${NVVS_PLUGINS};${LOCAL_NVVS_PLUGINS}" PARENT_SCOPE)

add_library(nvvs_plugins_common_objects OBJECT)
target_sources(nvvs_plugins_common_objects PRIVATE PluginCommon.cpp)
target_link_libraries(nvvs_plugins_common_objects PRIVATE nvvs_plugins_common_interface)
