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
find_package(fmt REQUIRED)

add_library(introspect_interface INTERFACE)
target_include_directories(introspect_interface INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(introspect_interface INTERFACE common_interface modules_interface)

add_library(dcgmmoduleintrospect SHARED)
define_dcgm_module(dcgmmoduleintrospect)
target_link_libraries(dcgmmoduleintrospect
    PRIVATE
        introspect_interface
        sdk_nvml_essentials_objects
        sdk_nvml_interface
)
target_sources(dcgmmoduleintrospect
    PRIVATE
    DcgmModuleIntrospect.cpp
    DcgmMetadataMgr.cpp
    DcgmModuleIntrospect.h
    dcgm_introspect_structs.h
    DcgmMetadataMgr.h
)
update_lib_ver(dcgmmoduleintrospect)
