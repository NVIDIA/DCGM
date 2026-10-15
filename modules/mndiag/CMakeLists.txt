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

add_subdirectory(adapters)
add_subdirectory(tests)

target_include_directories(mndiag_interface INTERFACE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/adapters")

target_link_libraries(mndiag_interface INTERFACE
    dcgm_ops
    dcgm_interface
    modules_interface
    nvvs_interface
    childprocess_interface)

target_sources(dcgmmodulemndiag_common
    PRIVATE
        dcgm_mndiag_structs.hpp
        MnDiagCommon.cpp
        MnDiagCommon.h
        MnDiagOutputHelpers.cpp
        MnDiagOutputHelpers.h)

target_link_libraries(dcgmmodulemndiag_common
    PRIVATE
        mndiag_interface
        dcgm_common
        Boost::regex
        fmt::fmt)

set(SRCS
    dcgm_mndiag_structs.hpp
    DcgmModuleMnDiag.cpp
    DcgmModuleMnDiag.h
    DcgmMnDiagManager.cpp
    DcgmMnDiagManager.h
    MnDiagCommon.cpp
    MnDiagCommon.h
    MnDiagOutputHelpers.cpp
    MnDiagOutputHelpers.h
    MpiRunner.cpp
    MpiRunner.h
    MnDiagMpiRunner.cpp
    MnDiagMpiRunner.h
    MnDiagTempDir.h
    MnDiagMpiMnubergemmRunner.cpp
    MnDiagMpiMnubergemmRunner.h
    MnDiagMpiNvloomRunner.cpp
    MnDiagMpiNvloomRunner.h
    MnDiagMpiNvbandwidthRunner.cpp
    MnDiagMpiNvbandwidthRunner.h
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
        nvbandwidth_common
        dcgm
        dcgm_common
        dcgm_logging
        dcgm_mutex
        serialize
        Boost::regex
        fmt::fmt)

target_sources(dcgmmodulemndiag_private_static
    PRIVATE
        ${SRCS})

define_dcgm_module(dcgmmodulemndiag)
target_link_libraries(dcgmmodulemndiag
    PRIVATE
        mndiag_interface
        childprocess
        remoteconn
        sdk_nvml_essentials_objects
        nvbandwidth_common
        dcgm
        dcgm_common
        dcgm_logging
        dcgm_mutex
        serialize
        jsoncpp_static
        Boost::regex
        fmt::fmt)

target_sources(dcgmmodulemndiag
    PRIVATE
        ${SRCS})

install(
    PROGRAMS cuda12/${CMAKE_SYSTEM_PROCESSOR}/mnubergemm
    DESTINATION "${DCGM_NVVS_PLUGINS_INSTALL_DIR}/cuda12/"
    COMPONENT Multinode_Cuda12)

install(CODE "
    file(RPATH_CHANGE
        FILE \"\$ENV{DESTDIR}$<INSTALL_PREFIX>/${CMAKE_INSTALL_LIBEXECDIR}/datacenter-gpu-manager-4/plugins/cuda12/mnubergemm\"
        OLD_RPATH \":::::::::::::::::::::::::::::::::::::::::::::::::::::\"
        NEW_RPATH \"$ORIGIN/../../cuda12/lib\")"
    COMPONENT Multinode_Cuda12)

install(
    PROGRAMS cuda12/${CMAKE_SYSTEM_PROCESSOR}/nvloom_cli
    DESTINATION "${CMAKE_INSTALL_LIBEXECDIR}/${PROJECT_NAME}/plugins/cuda12/"
    COMPONENT Multinode_Cuda12)

install(
    PROGRAMS cuda12/${CMAKE_SYSTEM_PROCESSOR}/mnnvbandwidth
    DESTINATION "${CMAKE_INSTALL_LIBEXECDIR}/${PROJECT_NAME}/plugins/cuda12/"
    COMPONENT Multinode_Cuda12)

install(
    PROGRAMS cuda13/${CMAKE_SYSTEM_PROCESSOR}/mnubergemm
    DESTINATION "${DCGM_NVVS_PLUGINS_INSTALL_DIR}/cuda13"
    COMPONENT Multinode_Cuda13)

install(CODE "
    file(RPATH_CHANGE
        FILE \"\$ENV{DESTDIR}$<INSTALL_PREFIX>/${CMAKE_INSTALL_LIBEXECDIR}/datacenter-gpu-manager-4/plugins/cuda13/mnubergemm\"
        OLD_RPATH \":::::::::::::::::::::::::::::::::::::::::::::::::::::\"
        NEW_RPATH \"$ORIGIN/../../cuda13/lib\")"
    COMPONENT Multinode_Cuda13)

install(
    PROGRAMS cuda13/${CMAKE_SYSTEM_PROCESSOR}/nvloom_cli
    DESTINATION "${CMAKE_INSTALL_LIBEXECDIR}/${PROJECT_NAME}/plugins/cuda13/"
    COMPONENT Multinode_Cuda13)

install(
    PROGRAMS cuda13/${CMAKE_SYSTEM_PROCESSOR}/mnnvbandwidth
    DESTINATION "${CMAKE_INSTALL_LIBEXECDIR}/${PROJECT_NAME}/plugins/cuda13"
    COMPONENT Multinode_Cuda13)

install(
    PROGRAMS cuda12/${CMAKE_SYSTEM_PROCESSOR}/mnubergemm
    DESTINATION ${DCGM_TESTS_INSTALL_DIR}/apps/nvvs/plugins/cuda12
    COMPONENT Tests)

install(
    PROGRAMS cuda13/${CMAKE_SYSTEM_PROCESSOR}/mnubergemm
    DESTINATION ${DCGM_TESTS_INSTALL_DIR}/apps/nvvs/plugins/cuda13
    COMPONENT Tests)

install(
    PROGRAMS cuda12/${CMAKE_SYSTEM_PROCESSOR}/nvloom_cli
    DESTINATION ${DCGM_TESTS_INSTALL_DIR}/apps/nvvs/plugins/cuda12
    COMPONENT Tests)

install(
    PROGRAMS cuda13/${CMAKE_SYSTEM_PROCESSOR}/nvloom_cli
    DESTINATION ${DCGM_TESTS_INSTALL_DIR}/apps/nvvs/plugins/cuda13
    COMPONENT Tests)

install(
    PROGRAMS cuda12/${CMAKE_SYSTEM_PROCESSOR}/mnnvbandwidth
    DESTINATION ${DCGM_TESTS_INSTALL_DIR}/apps/nvvs/plugins/cuda12
    COMPONENT Tests)

install(
    PROGRAMS cuda13/${CMAKE_SYSTEM_PROCESSOR}/mnnvbandwidth
    DESTINATION ${DCGM_TESTS_INSTALL_DIR}/apps/nvvs/plugins/cuda13
    COMPONENT Tests)

install(CODE "
    file(RPATH_CHANGE
        FILE \"\$ENV{DESTDIR}$<INSTALL_PREFIX>/${CMAKE_INSTALL_DATADIR}/dcgm_tests/apps/nvvs/plugins/cuda12/mnubergemm\"
        OLD_RPATH \":::::::::::::::::::::::::::::::::::::::::::::::::::::\"
        NEW_RPATH \"$ORIGIN\")

    file(RPATH_CHANGE
        FILE \"\$ENV{DESTDIR}$<INSTALL_PREFIX>/${CMAKE_INSTALL_DATADIR}/dcgm_tests/apps/nvvs/plugins/cuda13/mnubergemm\"
        OLD_RPATH \":::::::::::::::::::::::::::::::::::::::::::::::::::::\"
        NEW_RPATH \"$ORIGIN\")"
    COMPONENT Tests)
