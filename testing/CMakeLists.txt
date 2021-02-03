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
set(PYTHON_VER python)
#set(PYTHON_VER python3)

find_package(Jsoncpp REQUIRED)

add_executable(testdcgmunittests)
target_link_libraries(
    testdcgmunittests
    PRIVATE
        dcgm
        public_vgpu_interface
        dcgmmodulediag
        sdk_nvml_essentials_objects
        diag_interface
        module_common_objects
        module_common_interface
        ${JSONCPP_STATIC_LIBS}
)

target_sources(
    testdcgmunittests
    PRIVATE
    TestPolicyManager.cpp
    TestAllocator.cpp
    TestAllocator.h
    TestCacheManager.cpp
    TestCacheManager.h
    TestDcgmConnections.cpp
    TestDcgmConnections.h
    TestDcgmMutex.cpp
    TestDcgmMutex.h
    TestDiagManager.cpp
    TestDiagManager.h
    TestDiagManagerStrings.h
    TestDiagResponseWrapper.cpp
    TestDiagResponseWrapper.h
    TestFieldGroups.cpp
    TestFieldGroups.h
    TestGroupManager.cpp
    TestGroupManager.h
    TestHealthMonitor.cpp
    TestHealthMonitor.h
    TestKeyedVector.cpp
    TestKeyedVector.h
    TestDcgmModule.h
    TestDcgmUnitTests.cpp
    TestDcgmValue.cpp
    TestDcgmValue.h
    TestPolicyManager.h
    TestProtobuf.cpp
    TestProtobuf.h
    TestStatCollection.cpp
    TestStatCollection.h
    TestTopology.cpp
    TestTopology.h
    TestVersioning.cpp
    TestVersioning.h
)

add_subdirectory(stub)

install(DIRECTORY ${PYTHON_VER}/ DESTINATION ${DCGM_TESTS_INSTALL_DIR} COMPONENT Tests USE_SOURCE_PERMISSIONS)

set(BUILD_ID $ENV{BUILD_ID})
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${PYTHON_VER}/version.py.in ${CMAKE_CURRENT_BINARY_DIR}/version.py ESCAPE_QUOTES @ONLY)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/version.py DESTINATION ${DCGM_TESTS_INSTALL_DIR} COMPONENT Tests)

install(TARGETS testdcgmunittests stub_library_test RUNTIME DESTINATION ${DCGM_TESTS_APP_DIR} COMPONENT Tests)

install(PROGRAMS ${CMAKE_CURRENT_SOURCE_DIR}/run_tests.sh DESTINATION ${DCGM_TESTS_INSTALL_DIR} COMPONENT Tests)

# Install python bindings
install(
    FILES 
        ${CMAKE_CURRENT_SOURCE_DIR}/${PYTHON_VER}/common/__init__.py 
        ${CMAKE_CURRENT_SOURCE_DIR}/${PYTHON_VER}/common/dcgm_client_cli_parser.py
        ${CMAKE_CURRENT_SOURCE_DIR}/${PYTHON_VER}/common/dcgm_client_main.py
    DESTINATION
        ${DCGM_BINDINGS_INSTALL_DIR}/common
    COMPONENT DCGM)

install(
    FILES
        ${PYTHON_VER}/dcgm_agent.py
        ${PYTHON_VER}/dcgm_fields.py
        ${PYTHON_VER}/dcgm_structs.py
        ${PYTHON_VER}/dcgmvalue.py
        ${PYTHON_VER}/DcgmDiag.py
        ${PYTHON_VER}/DcgmGroup.py
        ${PYTHON_VER}/DcgmHandle.py
        ${PYTHON_VER}/DcgmJsonReader.py
        ${PYTHON_VER}/DcgmReader.py
        ${PYTHON_VER}/DcgmStatus.py
        ${PYTHON_VER}/DcgmSystem.py
        ${PYTHON_VER}/DcgmFieldGroup.py
        ${PYTHON_VER}/pydcgm.py
        ${PYTHON_VER}/dcgm_field_helpers.py
        ${PYTHON_VER}/dcgm_errors.py
        ${PYTHON_VER}/dcgm_collectd_plugin.py
        ${PYTHON_VER}/dcgm_fluentd.py
        ${PYTHON_VER}/dcgm_prometheus.py
        ${PYTHON_VER}/dcgm_telegraf.py
        ${PYTHON_VER}/blacklist_recommendations.py
    DESTINATION
        ${DCGM_BINDINGS_INSTALL_DIR}
    COMPONENT DCGM)
