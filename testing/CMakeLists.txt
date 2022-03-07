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

find_package(Jsoncpp REQUIRED)
find_package(Threads REQUIRED)
find_package(fmt REQUIRED)

add_executable(testdcgmunittests)
target_link_libraries(testdcgmunittests
    PRIVATE
        module_common_interface
        diag_interface

        module_common_objects
        dcgm_static_private
        modules_objects
        transport_objects
        dcgm_common
        dcgm_logging
        dcgm_mutex
        common_protobuf_objects
        sdk_nvml_essentials_objects
        sdk_nvml_loader
        dcgm
        dcgmmodulediag_private_static
        ${JSONCPP_STATIC_LIBS}
        ${CMAKE_THREAD_LIBS_INIT}
        fmt::fmt
        rt
        dl
)

target_sources(testdcgmunittests
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

target_link_options(testdcgmunittests PRIVATE -Wl,--version-script,${CMAKE_CURRENT_SOURCE_DIR}/unittests.linux_def)

add_subdirectory(stub)

install(DIRECTORY ${PYTHON_VER}/ DESTINATION ${DCGM_TESTS_INSTALL_DIR} COMPONENT Tests USE_SOURCE_PERMISSIONS)

set(BUILD_ID $ENV{BUILD_ID})
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${PYTHON_VER}/version.py.in ${CMAKE_CURRENT_BINARY_DIR}/version.py ESCAPE_QUOTES @ONLY)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/version.py DESTINATION ${DCGM_TESTS_INSTALL_DIR} COMPONENT Tests)

install(TARGETS testdcgmunittests stub_library_test RUNTIME DESTINATION ${DCGM_TESTS_APP_DIR} COMPONENT Tests)

install(PROGRAMS ${CMAKE_CURRENT_SOURCE_DIR}/run_tests.sh DESTINATION ${DCGM_TESTS_INSTALL_DIR} COMPONENT Tests)
install(PROGRAMS ${CMAKE_CURRENT_SOURCE_DIR}/generate_coverage_report.sh DESTINATION ${DCGM_TESTS_INSTALL_DIR} COMPONENT Tests)
install(PROGRAMS ${CMAKE_CURRENT_SOURCE_DIR}/process_coverage_report.sh DESTINATION ${DCGM_TESTS_INSTALL_DIR} COMPONENT Tests)
install(PROGRAMS ${CMAKE_CURRENT_SOURCE_DIR}/process_coverage_report.awk DESTINATION ${DCGM_TESTS_INSTALL_DIR} COMPONENT Tests)

# Install python bindings
set(DCGM_BINDINGS_PYTHON2_INSTALL_DIR ${DCGM_BINDINGS_INSTALL_DIR})
set(DCGM_BINDINGS_PYTHON3_INSTALL_DIR ${DCGM_BINDINGS_INSTALL_DIR}/python3)

install(
    FILES 
        ${CMAKE_CURRENT_SOURCE_DIR}/python/common/__init__.py
        ${CMAKE_CURRENT_SOURCE_DIR}/python/common/dcgm_client_cli_parser.py
        ${CMAKE_CURRENT_SOURCE_DIR}/python/common/dcgm_client_main.py
    DESTINATION
        ${DCGM_BINDINGS_PYTHON2_INSTALL_DIR}/common
    COMPONENT DCGM)

install(
    FILES
        python/dcgm_agent.py
        python/dcgm_fields.py
        python/dcgm_structs.py
        python/dcgmvalue.py
        python/DcgmDiag.py
        python/DcgmGroup.py
        python/DcgmHandle.py
        python/DcgmJsonReader.py
        python/DcgmReader.py
        python/DcgmStatus.py
        python/DcgmSystem.py
        python/DcgmFieldGroup.py
        python/pydcgm.py
        python/dcgm_field_helpers.py
        python/dcgm_errors.py
        python/dcgm_collectd_plugin.py
        python/dcgm_fluentd.py
        python/dcgm_prometheus.py
        python/dcgm_telegraf.py
        python/blacklist_recommendations.py
    DESTINATION
        ${DCGM_BINDINGS_PYTHON2_INSTALL_DIR}
    COMPONENT DCGM)

install(
    FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/python3/common/__init__.py
        ${CMAKE_CURRENT_SOURCE_DIR}/python3/common/dcgm_client_cli_parser.py
        ${CMAKE_CURRENT_SOURCE_DIR}/python3/common/dcgm_client_main.py
    DESTINATION
        ${DCGM_BINDINGS_PYTHON3_INSTALL_DIR}/common
    COMPONENT DCGM)

install(
    FILES
        python3/dcgm_agent.py
        python3/dcgm_fields.py
        python3/dcgm_fields_collectd.py
        python3/dcgm_structs.py
        python3/dcgmvalue.py
        python3/DcgmDiag.py
        python3/DcgmGroup.py
        python3/DcgmHandle.py
        python3/DcgmJsonReader.py
        python3/DcgmReader.py
        python3/DcgmStatus.py
        python3/DcgmSystem.py
        python3/DcgmFieldGroup.py
        python3/pydcgm.py
        python3/dcgm_field_helpers.py
        python3/dcgm_errors.py
        python3/dcgm_collectd_plugin.py
        python3/dcgm_fluentd.py
        python3/dcgm_prometheus.py
        python3/dcgm_telegraf.py
        python3/blacklist_recommendations.py
    DESTINATION
        ${DCGM_BINDINGS_PYTHON3_INSTALL_DIR}
    COMPONENT DCGM)

set(DCGM_COLLECTD_INSTALL_DIR "${CMAKE_INSTALL_DATAROOTDIR}/dcgm/collectd")
set(COLLECTD_TYPES_DB_GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/python3/dcgm_fields_collectd.py")
set(COLLECTD_TYPES_DB_FILE "${DCGM_COLLECTD_INSTALL_DIR}/types.db")

# For collectd types.db
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/types.db
  DEPENDS ${COLLECTD_TYPES_DB_GENERATOR}
  # -B to prevent caching bytecode
  COMMAND ${PYTHON_VER} -B ${COLLECTD_TYPES_DB_GENERATOR} >${CMAKE_CURRENT_BINARY_DIR}/types.db)

# For collectd types.db
add_custom_target(
  db_types ALL
  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/types.db)

# For collectd types.db
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/types.db DESTINATION ${DCGM_COLLECTD_INSTALL_DIR} COMPONENT DCGM)

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/python3/dcgm-collectd-example.conf DESTINATION ${DCGM_COLLECTD_INSTALL_DIR} COMPONENT DCGM)



