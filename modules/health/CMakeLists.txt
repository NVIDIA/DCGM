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
add_library(health_interface INTERFACE)
target_include_directories(health_interface INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(health_interface INTERFACE dcgm_interface modules_interface)

add_library(health_objects OBJECT)
target_link_libraries(health_objects PUBLIC health_interface)
target_compile_options(health_objects PRIVATE -fPIC)
target_sources(
    health_objects
    PRIVATE
    DcgmHealthWatch.h
    DcgmModuleHealth.h
    dcgm_health_structs.h
    DcgmHealthResponse.h
    DcgmModuleHealth.cpp
    DcgmHealthWatch.cpp
    DcgmHealthResponse.cpp
)

add_library(dcgmmodulehealth SHARED)
target_compile_options(dcgmmodulehealth PRIVATE -fPIC)
dcgm_macro_modules_link_options(dcgmmodulehealth)
target_link_libraries(
    dcgmmodulehealth
    PRIVATE
        health_objects
        modules_objects
        sdk_nvml_essentials_objects
        dcgm
)
update_lib_ver(dcgmmodulehealth)
