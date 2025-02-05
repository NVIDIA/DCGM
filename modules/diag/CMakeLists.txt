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

target_include_directories(diag_interface INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(diag_interface INTERFACE dcgm_interface modules_interface nvvs_interface config_interface childprocess_interface)

target_sources(dcgmmodulediag_common
    PRIVATE
        DcgmDiagCommon.cpp
        DcgmDiagCommon.h
)
target_link_libraries(dcgmmodulediag_common
    PRIVATE
        diag_interface
        dcgm_common
        fmt::fmt
)

set(SRCS
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

target_link_libraries(dcgmmodulediag_private_static
    PRIVATE
        diag_interface
        transport_interface
        childprocess
        dcgm_common
        serialize
        fmt::fmt
)
target_sources(dcgmmodulediag_private_static
    PRIVATE
        ${SRCS}
)

define_dcgm_module(dcgmmodulediag)
target_link_libraries(dcgmmodulediag
    PRIVATE
        dcgmmoduleconfig
        diag_interface
        config_interface
        childprocess
        sdk_nvml_essentials_objects
        dcgm
        dcgm_common
        dcgm_logging
        dcgm_mutex
        serialize
        ${JSONCPP_STATIC_LIBS}
        fmt::fmt
)
target_sources(dcgmmodulediag
    PRIVATE
        ${SRCS}
)
