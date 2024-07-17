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
declare_nvvs_plugin(common .)

add_library(pluginCommon SHARED)
nvvs_plugin_srcs(pluginCommon)
set_target_properties(pluginCommon PROPERTIES LIBRARY_OUTPUT_NAME "pluginCommon")
update_lib_ver(pluginCommon)
install(TARGETS pluginCommon
    LIBRARY DESTINATION ${DCGM_NVVS_PLUGINS_INSTALL_DIR}/cudaless
    PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
    COMPONENT DCGM)
set_target_properties(pluginCommon PROPERTIES INSTALL_TO_DIR ${DCGM_NVVS_PLUGINS_INSTALL_DIR}/cudaless)

set(NVVS_PLUGINS "${NVVS_PLUGINS};pluginCommon" PARENT_SCOPE)

add_library(nvvs_plugins_common_objects OBJECT)
target_sources(nvvs_plugins_common_objects PRIVATE PluginCommon.cpp)
target_link_libraries(nvvs_plugins_common_objects PRIVATE nvvs_plugins_common_interface)

