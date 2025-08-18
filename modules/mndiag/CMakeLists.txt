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

target_include_directories(mndiag_interface INTERFACE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/adapters")

target_link_libraries(mndiag_interface INTERFACE
    dcgm_interface
    modules_interface
    nvvs_interface
    config_interface
    childprocess_interface)

target_sources(dcgmmodulemndiag_common
    PRIVATE
        dcgm_mndiag_structs.hpp
        MnDiagCommon.cpp
        MnDiagCommon.h)

target_link_libraries(dcgmmodulemndiag_common
    PRIVATE
        mndiag_interface
        dcgm_common
        fmt::fmt)

set(SRCS
    dcgm_mndiag_structs.hpp
    DcgmModuleMnDiag.cpp
    DcgmModuleMnDiag.h
    DcgmMnDiagManager.cpp
    DcgmMnDiagManager.h
    MnDiagCommon.cpp
    MnDiagCommon.h
    MpiRunner.cpp
    MpiRunner.h
    MnDiagMpiRunner.cpp
    MnDiagMpiRunner.h
    MnDiagStateMachine.cpp
    MnDiagStateMachine.h
    MnDiagProcessUtils.cpp
    MnDiagProcessUtils.h)

target_link_libraries(dcgmmodulemndiag_private_static
    PRIVATE
        mndiag_interface
        transport_interface
        childprocess
        remoteconn
        sdk_nvml_essentials_objects
        dcgm
        dcgm_common
        dcgm_logging
        dcgm_mutex
        serialize
        fmt::fmt)

target_sources(dcgmmodulemndiag_private_static
    PRIVATE
        ${SRCS})

define_dcgm_module(dcgmmodulemndiag)
target_link_libraries(dcgmmodulemndiag
    PRIVATE
        dcgmmoduleconfig
        mndiag_interface
        config_interface
        childprocess
        remoteconn
        sdk_nvml_essentials_objects
        dcgm
        dcgm_common
        dcgm_logging
        dcgm_mutex
        serialize
        ${JSONCPP_STATIC_LIBS}
        fmt::fmt)

target_sources(dcgmmodulemndiag
    PRIVATE
        ${SRCS})