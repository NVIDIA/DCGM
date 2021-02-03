# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

add_library(dcgm SHARED)
update_lib_ver(dcgm)

target_link_options(
    dcgm
    PRIVATE
    -Xlinker
    --version-script=${CMAKE_CURRENT_SOURCE_DIR}/../dcgmlib.linux_def
)

target_link_options(dcgm PRIVATE -rdynamic)

target_link_libraries(dcgm PRIVATE rt dl)

target_include_directories(dcgm PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(
    dcgm
    PRIVATE
        buildinfo_objects
        common_objects
        common_protobuf_objects
        config_interface
        dcgm_interface
        diag_interface
        health_interface
        introspect_interface
        modules_interface
        modules_objects
        module_common_interface
        module_common_objects
        nvswitch_interface
        policy_interface
        public_profiling_interface
        public_vgpu_interface
        sdk_nvml_essentials_objects
        sdk_nvml_interface
        sdk_nvml_loader
        transport_objects
)

target_sources(
    dcgm
    PRIVATE
    DcgmCacheManager.cpp
    DcgmFieldGroup.cpp
    DcgmVersion.cpp
    DcgmApi.cpp
    DcgmClientHandler.cpp
    DcgmGroupManager.cpp
    DcgmHostEngineHandler.cpp
    DcgmGpuInstance.cpp
    DcgmCoreCommunication.cpp
    DcgmMigManager.cpp
    DcgmMigTypes.cpp
    dcgm.c
)

add_library(dcgmtest_interface INTERFACE)
target_include_directories(dcgmtest_interface INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
