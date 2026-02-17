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

install(CODE "
    file(INSTALL
        DESTINATION \"$<INSTALL_PREFIX>/${CMAKE_INSTALL_LIBEXECDIR}/${PROJECT_NAME}/plugins/cuda12\"
        TYPE SHARED_LIBRARY
        FILES \"${CMAKE_CURRENT_SOURCE_DIR}/cuda12/${CMAKE_SYSTEM_PROCESSOR}/libcurand.so.10.3.10.19\")

    execute_process(
        COMMAND \"${CMAKE_COMMAND}\" -E create_symlink libcurand.so.10.3.10.19 libcurand.so.10
        WORKING_DIRECTORY \"\$ENV{DESTDIR}$<INSTALL_PREFIX>/${CMAKE_INSTALL_LIBEXECDIR}/${PROJECT_NAME}/plugins/cuda12\"
        COMMAND_ERROR_IS_FATAL ANY)

    execute_process(
        COMMAND \"${CMAKE_COMMAND}\" -E create_symlink libcurand.so.10 libcurand.so
        WORKING_DIRECTORY \"\$ENV{DESTDIR}$<INSTALL_PREFIX>/${CMAKE_INSTALL_LIBEXECDIR}/${PROJECT_NAME}/plugins/cuda12\"
        COMMAND_ERROR_IS_FATAL ANY)

    execute_process(
        COMMAND \"${CMAKE_COMMAND}\" -E make_directory updated
        WORKING_DIRECTORY \"\$ENV{DESTDIR}$<INSTALL_PREFIX>/${CMAKE_INSTALL_LIBEXECDIR}/${PROJECT_NAME}/plugins/cuda12\"
        COMMAND_ERROR_IS_FATAL ANY)
 
    execute_process(
        COMMAND \"${CMAKE_COMMAND}\" -E create_symlink ../libcurand.so.10.3.10.19 libcurand.so.10
        WORKING_DIRECTORY \"\$ENV{DESTDIR}$<INSTALL_PREFIX>/${CMAKE_INSTALL_LIBEXECDIR}/${PROJECT_NAME}/plugins/cuda12/updated\"
        COMMAND_ERROR_IS_FATAL ANY)"
    COMPONENT Cuda12)

install(
    PROGRAMS cuda12/${CMAKE_SYSTEM_PROCESSOR}/mnubergemm
    DESTINATION "${CMAKE_INSTALL_LIBEXECDIR}/${PROJECT_NAME}/plugins/cuda12/"
    COMPONENT Multinode_Cuda12)

install(CODE "
    file(RPATH_CHANGE
        FILE \"\$ENV{DESTDIR}$<INSTALL_PREFIX>/${CMAKE_INSTALL_LIBEXECDIR}/datacenter-gpu-manager-4/plugins/cuda12/mnubergemm\"
        OLD_RPATH \"::::::::::::::::::::::::::::::::::::::::::::::::::::\"
        NEW_RPATH \"$ORIGIN:$ORIGIN/../../../../${CMAKE_INSTALL_LIBDIR}\")"
    COMPONENT Multinode_Cuda12)

install(CODE "
    file(INSTALL
        DESTINATION \"$<INSTALL_PREFIX>/${CMAKE_INSTALL_LIBEXECDIR}/${PROJECT_NAME}/plugins/cuda13\"
        TYPE SHARED_LIBRARY
        FILES \"${CMAKE_CURRENT_SOURCE_DIR}/cuda13/${CMAKE_SYSTEM_PROCESSOR}/libcurand.so.10.4.0.35\")

    execute_process(
        COMMAND \"${CMAKE_COMMAND}\" -E create_symlink libcurand.so.10.4.0.35 libcurand.so.10
        WORKING_DIRECTORY \"\$ENV{DESTDIR}$<INSTALL_PREFIX>/${CMAKE_INSTALL_LIBEXECDIR}/${PROJECT_NAME}/plugins/cuda13\"
        COMMAND_ERROR_IS_FATAL ANY)

    execute_process(
        COMMAND \"${CMAKE_COMMAND}\" -E create_symlink libcurand.so.10 libcurand.so
        WORKING_DIRECTORY \"\$ENV{DESTDIR}$<INSTALL_PREFIX>/${CMAKE_INSTALL_LIBEXECDIR}/${PROJECT_NAME}/plugins/cuda13\"
        COMMAND_ERROR_IS_FATAL ANY)

    execute_process(
        COMMAND \"${CMAKE_COMMAND}\" -E make_directory updated
        WORKING_DIRECTORY \"\$ENV{DESTDIR}$<INSTALL_PREFIX>/${CMAKE_INSTALL_LIBEXECDIR}/${PROJECT_NAME}/plugins/cuda13\"
        COMMAND_ERROR_IS_FATAL ANY)

    execute_process(
        COMMAND \"${CMAKE_COMMAND}\" -E create_symlink ../libcurand.so.10.4.0.35 libcurand.so.10
        WORKING_DIRECTORY \"\$ENV{DESTDIR}$<INSTALL_PREFIX>/${CMAKE_INSTALL_LIBEXECDIR}/${PROJECT_NAME}/plugins/cuda13/updated\"
        COMMAND_ERROR_IS_FATAL ANY)"
    COMPONENT Cuda13)

install(
    PROGRAMS cuda13/${CMAKE_SYSTEM_PROCESSOR}/mnubergemm
    DESTINATION "${CMAKE_INSTALL_LIBEXECDIR}/${PROJECT_NAME}/plugins/cuda13"
    COMPONENT Multinode_Cuda13)

install(CODE "
    file(RPATH_CHANGE
        FILE \"\$ENV{DESTDIR}$<INSTALL_PREFIX>/libexec/datacenter-gpu-manager-4/plugins/cuda13/mnubergemm\"
        OLD_RPATH \"::::::::::::::::::::::::::::::::::::::::::::::::::::\"
        NEW_RPATH \"$ORIGIN:$ORIGIN/../../../../${CMAKE_INSTALL_LIBDIR}\")"
    COMPONENT Multinode_Cuda13)

install(
    PROGRAMS cuda12/${CMAKE_SYSTEM_PROCESSOR}/mnubergemm
    DESTINATION ${DCGM_TESTS_INSTALL_DIR}/apps/nvvs/plugins/cuda12
    COMPONENT Tests)

install(
    PROGRAMS cuda13/${CMAKE_SYSTEM_PROCESSOR}/mnubergemm
    DESTINATION ${DCGM_TESTS_INSTALL_DIR}/apps/nvvs/plugins/cuda13
    COMPONENT Tests)

install(CODE "
    file(INSTALL
        DESTINATION \"$<INSTALL_PREFIX>/${CMAKE_INSTALL_DATADIR}/dcgm_tests/apps/nvvs/plugins/cuda12\"
        TYPE SHARED_LIBRARY
        FILES \"${CMAKE_CURRENT_SOURCE_DIR}/cuda12/${CMAKE_SYSTEM_PROCESSOR}/libcurand.so.10.3.10.19\")

    file(INSTALL
        DESTINATION \"$<INSTALL_PREFIX>/${CMAKE_INSTALL_DATADIR}/dcgm_tests/apps/nvvs/plugins/cuda13\"
        TYPE SHARED_LIBRARY
        FILES \"${CMAKE_CURRENT_SOURCE_DIR}/cuda13/${CMAKE_SYSTEM_PROCESSOR}/libcurand.so.10.4.0.35\")

    file(RPATH_CHANGE
        FILE \"\$ENV{DESTDIR}$<INSTALL_PREFIX>/${CMAKE_INSTALL_DATADIR}/dcgm_tests/apps/nvvs/plugins/cuda12/mnubergemm\"
        OLD_RPATH \"::::::::::::::::::::::::::::::::::::::::::::::::::::\"
        NEW_RPATH \"$ORIGIN:$ORIGIN/../../../${DCGM_TESTS_ARCH}\")

    file(RPATH_CHANGE
        FILE \"\$ENV{DESTDIR}$<INSTALL_PREFIX>/${CMAKE_INSTALL_DATADIR}/dcgm_tests/apps/nvvs/plugins/cuda13/mnubergemm\"
        OLD_RPATH \"::::::::::::::::::::::::::::::::::::::::::::::::::::\"
        NEW_RPATH \"$ORIGIN:$ORIGIN/../../../${DCGM_TESTS_ARCH}\")

    execute_process(
        COMMAND \"${CMAKE_COMMAND}\" -E create_symlink libcurand.so.10.3.10.19 libcurand.so.10
        COMMAND \"${CMAKE_COMMAND}\" -E make_directory updated
        WORKING_DIRECTORY \"\$ENV{DESTDIR}$<INSTALL_PREFIX>/${CMAKE_INSTALL_DATADIR}/dcgm_tests/apps/nvvs/plugins/cuda12\"
        COMMAND_ERROR_IS_FATAL ANY)

    execute_process(
        COMMAND \"${CMAKE_COMMAND}\" -E create_symlink libcurand.so.10 libcurand.so
        WORKING_DIRECTORY \"\$ENV{DESTDIR}$<INSTALL_PREFIX>/${CMAKE_INSTALL_DATADIR}/dcgm_tests/apps/nvvs/plugins/cuda12\"
        COMMAND_ERROR_IS_FATAL ANY)

    execute_process(
        COMMAND \"${CMAKE_COMMAND}\" -E create_symlink ../libcurand.so.10.3.10.19 libcurand.so.10
        WORKING_DIRECTORY \"\$ENV{DESTDIR}$<INSTALL_PREFIX>/${CMAKE_INSTALL_DATADIR}/dcgm_tests/apps/nvvs/plugins/cuda12/updated\"
        COMMAND_ERROR_IS_FATAL ANY)

    execute_process(
        COMMAND \"${CMAKE_COMMAND}\" -E create_symlink libcurand.so.10.4.0.35 libcurand.so.10
        COMMAND \"${CMAKE_COMMAND}\" -E make_directory updated
        WORKING_DIRECTORY \"\$ENV{DESTDIR}$<INSTALL_PREFIX>/${CMAKE_INSTALL_DATADIR}/dcgm_tests/apps/nvvs/plugins/cuda13\"
        COMMAND_ERROR_IS_FATAL ANY)

    execute_process(
        COMMAND \"${CMAKE_COMMAND}\" -E create_symlink libcurand.so.10 libcurand.so
        WORKING_DIRECTORY \"\$ENV{DESTDIR}$<INSTALL_PREFIX>/${CMAKE_INSTALL_DATADIR}/dcgm_tests/apps/nvvs/plugins/cuda13\"
        COMMAND_ERROR_IS_FATAL ANY)

    execute_process(
        COMMAND \"${CMAKE_COMMAND}\" -E create_symlink ../libcurand.so.10.4.0.35 libcurand.so.10
        WORKING_DIRECTORY \"\$ENV{DESTDIR}$<INSTALL_PREFIX>/${CMAKE_INSTALL_DATADIR}/dcgm_tests/apps/nvvs/plugins/cuda13/updated\"
        COMMAND_ERROR_IS_FATAL ANY)"
    COMPONENT Tests)
