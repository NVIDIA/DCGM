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
add_library(policy_interface INTERFACE)
target_include_directories(policy_interface INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(policy_interface INTERFACE dcgm_interface modules_interface)

add_library(policy_objects OBJECT)
target_link_libraries(policy_objects PUBLIC policy_interface)
target_compile_options(policy_objects PRIVATE -fPIC)
target_sources(policy_objects PRIVATE
    dcgm_policy_structs.h
    DcgmModulePolicy.h
    DcgmPolicyManager.cpp
    DcgmPolicyManager.h
    DcgmModulePolicy.cpp
    )

add_library(dcgmmodulepolicy SHARED)
dcgm_macro_modules_link_options(dcgmmodulepolicy)
target_compile_options(dcgmmodulepolicy PRIVATE -fPIC)
target_link_libraries(dcgmmodulepolicy PRIVATE policy_objects modules_objects sdk_nvml_essentials_objects dcgm)
update_lib_ver(dcgmmodulepolicy)
