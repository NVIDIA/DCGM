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

add_executable(testdcgmunittests)

target_link_libraries(testdcgmunittests
    PRIVATE
        ${CMAKE_THREAD_LIBS_INIT}
        ${JSONCPP_STATIC_LIBS}
        dcgm_common
        dcgm_logging
        dcgm_mutex
        dcgm_static_private
        dcgmmodulediag_private_static
        diag_interface
        dl
        fmt::fmt
        module_common_interface
        module_common_objects
        modules_objects
        dcgm
        rt
        sdk_nvml_essentials_objects
        sdk_nvml_loader
        serialize
        transport_objects)

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
        TestTopology.cpp
        TestTopology.h
        TestVersioning.cpp
        TestVersioning.h)

target_compile_definitions(testdcgmunittests PRIVATE -DTEST_DCGMCACHEMANAGER)

target_link_options(testdcgmunittests PRIVATE
    "-Wl,--version-script,${CMAKE_CURRENT_SOURCE_DIR}/unittests.linux_def")

add_subdirectory(stub)

configure_file(python3/version.py.in version.py ESCAPE_QUOTES @ONLY)

install(
    FILES "${CMAKE_CURRENT_BINARY_DIR}/version.py"
    DESTINATION "${DCGM_TESTS_INSTALL_DIR}"
    COMPONENT Tests)

install(
    DIRECTORY python3/
    DESTINATION "${DCGM_TESTS_INSTALL_DIR}"
    COMPONENT Tests
    FILES_MATCHING
        PATTERN "*.conf"
        PATTERN "*.json"
        PATTERN "*.py"
        PATTERN ".coveragerc"
        PATTERN "LICENSE.txt"
        PATTERN "pylintrc")

if (CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
    install(
        FILES
            "python3/apps/cuda_ctx_create/busy_gpu64.ptx"
            "python3/apps/cuda_ctx_create/cuda_assert.ptx"
        DESTINATION "${DCGM_TESTS_INSTALL_DIR}/apps/cuda_ctx_create/"
        COMPONENT Tests)

    install(
        PROGRAMS
            "python3/apps/cuda_ctx_create/cuda_assert_64bit"
            "python3/apps/cuda_ctx_create/cuda_ctx_create_64bit"
        DESTINATION "${DCGM_TESTS_INSTALL_DIR}/apps/cuda_ctx_create/"
        COMPONENT Tests)

    install(PROGRAMS "python3/apps/nvswitch-audit/nvswitch-audit"
        DESTINATION "${DCGM_TESTS_INSTALL_DIR}/apps/nvswitch-audit/"
        COMPONENT Tests)

    install(PROGRAMS "python3/apps/p2p_bandwidth/p2p_bandwidth"
        DESTINATION "${DCGM_TESTS_INSTALL_DIR}/apps/p2p_bandwidth"
        COMPONENT Tests)

    install(PROGRAMS "python3/apps/nvpex2/nvpex2"
        DESTINATION "${DCGM_TESTS_INSTALL_DIR}/apps/nvpex2"
        COMPONENT Tests)
elseif (CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
    install(
        FILES
            "python3/apps/cuda_ctx_create/busy_gpu_aarch64.ptx"
            "python3/apps/cuda_ctx_create/cuda_assert.ptx"
        DESTINATION "${DCGM_TESTS_INSTALL_DIR}/apps/cuda_ctx_create/"
        COMPONENT Tests)

    install(
        PROGRAMS
            "python3/apps/cuda_ctx_create/cuda_assert_aarch64"
            "python3/apps/cuda_ctx_create/cuda_ctx_create_aarch64"
        DESTINATION "${DCGM_TESTS_INSTALL_DIR}/apps/cuda_ctx_create/"
        COMPONENT Tests)
endif()

install(
    TARGETS testdcgmunittests
    RUNTIME
        DESTINATION "${CMAKE_INSTALL_DATADIR}/dcgm_tests/apps/${DCGM_TESTS_ARCH}"
        COMPONENT Tests)

install(
    PROGRAMS
        run_tests.sh
        python3/test_times.sh
        generate_coverage_report.sh
        process_coverage_report.sh
        process_coverage_report.awk
        gen_validation.sh
        test_validation.sh
    DESTINATION "${DCGM_TESTS_INSTALL_DIR}"
    COMPONENT Tests)

# Install python bindings
install(
    FILES
        python3/dcgm_agent.py
        python3/dcgm_fields.py
        python3/dcgm_fields_collectd.py
        python3/dcgm_fields_internal.py
        python3/dcgm_structs.py
        python3/dcgmvalue.py
        python3/DcgmDiag.py
        python3/DcgmMnDiag.py
        python3/DcgmGroup.py
        python3/DcgmHandle.py
        python3/DcgmJsonReader.py
	    python3/DcgmLogging.py
        python3/DcgmReader.py
        python3/DcgmStatus.py
        python3/DcgmSystem.py
        python3/DcgmFieldGroup.py
        python3/pydcgm.py
        python3/dcgm_field_helpers.py
        python3/dcgm_errors.py
        python3/dcgm_collectd_plugin.py
        python3/dcgm_fluentd.py
        python3/dcgm_telegraf.py
        python3/denylist_recommendations.py
    DESTINATION "${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME}/bindings/python3"
    COMPONENT Core)

install(
    FILES
        python3/common/__init__.py
        python3/common/dcgm_client_cli_parser.py
        python3/common/dcgm_client_main.py
    DESTINATION "${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME}/bindings/python3/common"
    COMPONENT Core)

set(COLLECTD_TYPES_DB_GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/python3/dcgm_fields_collectd.py")

# For collectd types.db
add_custom_command(
  OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/types.db"
  DEPENDS ${COLLECTD_TYPES_DB_GENERATOR}
  # -B to prevent caching bytecode
  COMMAND Python3::Interpreter -B "${COLLECTD_TYPES_DB_GENERATOR}" > "${CMAKE_CURRENT_BINARY_DIR}/types.db")

# For collectd types.db
add_custom_target(
  db_types ALL
  DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/types.db")

# For collectd types.db
install(
    FILES "${CMAKE_CURRENT_BINARY_DIR}/types.db"
    DESTINATION "${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/collectd"
    COMPONENT Core)

install(
    FILES python3/dcgm-collectd-example.conf
    DESTINATION "${CMAKE_INSTALL_DOCDIR}/examples/"
    COMPONENT Core)
