# Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
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

target_include_directories(policy_interface INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(policy_interface INTERFACE dcgm_interface modules_interface)

target_sources(dcgmmodulepolicy_common
    PRIVATE
        DcgmPolicyUtils.cpp
        DcgmPolicyUtils.h)
target_link_libraries(dcgmmodulepolicy_common PRIVATE policy_interface dcgm_logging)

define_dcgm_module(dcgmmodulepolicy)
target_link_libraries(dcgmmodulepolicy
    PRIVATE
        dcgmmodulepolicy_common
        policy_interface
        dcgm_thread
        sdk_nvml_essentials_objects
        jsoncpp_static
        yaml-cpp::yaml-cpp
)
target_sources(dcgmmodulepolicy
    PRIVATE
        dcgm_policy_structs.h
        DcgmModulePolicy.cpp
        DcgmModulePolicy.h
        DcgmPolicyManager.cpp
        DcgmPolicyManager.h
        DcgmPolicy.cpp
        DcgmPolicy.h
        DcgmPolicyRegistry.cpp
        DcgmPolicyRegistry.h
        DcgmPolicyEvaluator.cpp
        DcgmPolicyEvaluator.h
        DcgmPolicyNotificationRouter.cpp
        DcgmPolicyNotificationRouter.h
        DcgmPolicyCallbackRAII.cpp
        DcgmPolicyCallbackRAII.h
        DcgmPolicyPersistent.cpp
        DcgmPolicyPersistent.h
)

add_subdirectory(tests)
