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

set(SRCS 
    DcgmCMUtils.cpp
    DcgmCacheManager.cpp
    DcgmFieldGroup.cpp
    DcgmVersion.cpp
    DcgmApi.cpp
    DcgmClientHandler.cpp
    DcgmGroupManager.cpp
    DcgmHostEngineHandler.cpp
    DcgmInjectionNvmlManager.cpp
    DcgmGpuInstance.cpp
    DcgmCoreCommunication.cpp
    DcgmMigManager.cpp
    DcgmProfiles.cpp
    DcgmTopology.cpp
    DcgmGpmManager.cpp
    DcgmVgpu.cpp
    DcgmKmsgReader.cpp
    dcgm.c
    dcgm_errors.c
    dcgm_fields.cpp
    dcgm_helpers.c
    DcgmResourceManager.cpp
)

set(INCLUDES "${CMAKE_CURRENT_SOURCE_DIR}")

set(DEPS 
    dcgm_interface
    common_interface
    dcgm_thread_interface
    config_interface
    diag_interface
    mndiag_interface
    health_interface
    introspect_interface
    module_common_interface
    modules_interface
    nvswitch_interface
    policy_interface
    public_profiling_interface
    public_vgpu_interface
    sdk_nvml_interface
    sdk_nvml_loader
    sysmon_interface
    transport_interface
    fmt::fmt
)

target_sources(dcgm_entity_types PRIVATE DcgmEntityTypes.cpp)
target_include_directories(dcgm_entity_types PRIVATE "${INCLUDES}")
target_link_libraries(dcgm_entity_types PRIVATE ${DEPS})

target_sources(dcgm_static_private PRIVATE ${SRCS})
target_include_directories(dcgm_static_private PRIVATE "${INCLUDES}")
target_link_libraries(dcgm_static_private PRIVATE ${DEPS})
target_link_libraries(dcgm_static_private PRIVATE dcgm_entity_types)

target_sources(dcgm PRIVATE ${SRCS})
target_include_directories(dcgm PRIVATE "${INCLUDES}")
target_compile_definitions(dcgm PRIVATE DCGM_API_EXPORT=1)
target_link_options(dcgm PRIVATE -Wl,--version-script,${CMAKE_CURRENT_SOURCE_DIR}/../dcgmlib.linux_def)

target_link_libraries(dcgm
    INTERFACE
        dcgm_interface
    PRIVATE
        ${DEPS}
        modules_objects
        dcgm_entity_types
        -Wl,--whole-archive
        buildinfo_objects
        module_common_objects
        transport_objects
        dcgm_common
        dcgm_logging
        dcgm_mutex
        dcgm_thread
        -Wl,--no-whole-archive
        transport_objects
        sdk_nvml_essentials_objects
        sdk_nvml_loader
        rt
        dl
)

target_include_directories(dcgmtest_interface INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
