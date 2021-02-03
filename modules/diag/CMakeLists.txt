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
add_library(diag_interface INTERFACE)
target_include_directories(diag_interface INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(diag_interface INTERFACE dcgm_interface modules_interface nvvs_interface config_interface)

add_library(diag_objects OBJECT)
target_link_libraries(diag_objects PUBLIC diag_interface dcgmmoduleconfig config_interface)
target_sources(
    diag_objects
    PRIVATE
    DcgmDiagCommon.cpp
    DcgmDiagCommon.h
    DcgmDiagManager.cpp
    DcgmDiagManager.h
    DcgmDiagResponseWrapper.cpp
    DcgmDiagResponseWrapper.h
    dcgm_diag_structs.h
    DcgmModuleDiag.cpp
    DcgmModuleDiag.h
)

find_package(Jsoncpp REQUIRED)

add_library(dcgmmodulediag SHARED)
dcgm_macro_modules_link_options(dcgmmodulediag)
target_link_libraries(dcgmmodulediag PUBLIC dcgmmoduleconfig)
target_link_libraries(
    dcgmmodulediag
    PRIVATE
        diag_objects
        modules_objects
        sdk_nvml_essentials_objects
        dcgm
)
target_link_libraries(dcgmmodulediag PRIVATE ${JSONCPP_STATIC_LIBS})
update_lib_ver(dcgmmodulediag)
