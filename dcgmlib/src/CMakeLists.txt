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
add_subdirectory(tests)

find_package(fmt REQUIRED)

set(SRCS
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
    dcgm.c
    dcgm_errors.c
    dcgm_fields.cpp
)

set(INCLUDES
    ${CMAKE_CURRENT_SOURCE_DIR}
)

set(DEPS 
    dcgm_interface
    common_interface
    config_interface
    diag_interface
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
    transport_interface
    fmt::fmt
)

add_library(dcgm_mig_types STATIC)
target_sources(dcgm_mig_types PRIVATE DcgmMigTypes.cpp)
target_include_directories(dcgm_mig_types PRIVATE ${INCLUDES})
target_link_libraries(dcgm_mig_types PRIVATE ${DEPS})

add_library(dcgm_static_private STATIC)
target_sources(dcgm_static_private PRIVATE ${SRCS})
target_include_directories(dcgm_static_private PRIVATE ${INCLUDES})
target_link_libraries(dcgm_static_private PRIVATE ${DEPS})
target_link_libraries(dcgm_static_private PRIVATE dcgm_mig_types)

# add_library(dcgm_static_public STATIC)
# target_compile_definitions(dcgm_static_public PUBLIC DCGM_API_EXPORT=1)
# target_sources(dcgm_static_public PRIVATE ${SRCS})
# target_include_directories(dcgm_static_public PRIVATE ${INCLUDES})
# target_link_libraries(dcgm_static_public PRIVATE ${DEPS})

add_library(dcgm SHARED)
update_lib_ver(dcgm)


target_sources(dcgm PRIVATE ${SRCS})
target_include_directories(dcgm PRIVATE ${INCLUDES})
target_compile_definitions(dcgm PRIVATE DCGM_API_EXPORT=1)
target_link_options(dcgm PRIVATE -Wl,--version-script,${CMAKE_CURRENT_SOURCE_DIR}/../dcgmlib.linux_def)

target_link_libraries(
    dcgm
    PRIVATE
        ${DEPS}
        modules_objects
        dcgm_mig_types
        -Wl,--whole-archive
        buildinfo_objects
        common_protobuf_objects
        module_common_objects
        transport_objects
        dcgm_common
        dcgm_logging
        dcgm_mutex
        -Wl,--no-whole-archive
        transport_objects
        sdk_nvml_essentials_objects
        sdk_nvml_loader
        rt
        dl
)

add_library(dcgmtest_interface INTERFACE)
target_include_directories(dcgmtest_interface INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
