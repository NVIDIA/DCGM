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

if (NOT BUILD_TESTING)
    return()
endif()

# Keep the dynamic test and fake libdcgm.so.4 in one directory so $ORIGIN finds
# the fake library without LD_LIBRARY_PATH.
set(DCGM_STUB_DYNAMIC_TEST_DIR "${CMAKE_CURRENT_BINARY_DIR}/dynamic")

add_library(dcgm_stub_fake_dcgm SHARED)
target_sources(dcgm_stub_fake_dcgm PRIVATE FakeDcgmDynamicLib.c)
target_link_libraries(dcgm_stub_fake_dcgm PRIVATE dcgm_interface)
set_target_properties(dcgm_stub_fake_dcgm PROPERTIES
    OUTPUT_NAME dcgm
    SOVERSION ${PROJECT_VERSION_MAJOR}
    LIBRARY_OUTPUT_DIRECTORY "${DCGM_STUB_DYNAMIC_TEST_DIR}"
)

add_executable(dcgm_stub_tests)
target_sources(dcgm_stub_tests
    PRIVATE
        test_maps_tokens.cpp
        DcgmStubStaticTests.cpp)
target_link_libraries(dcgm_stub_tests
    PRIVATE
        Catch2::Catch2WithMain
        dcgm_interface
        dcgm_stub
        nvmli_interface
        test-utils)

if (CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
    catch_discover_tests(dcgm_stub_tests EXTRA_ARGS --colour-mode ansi)
endif()

add_executable(dcgm_stub_dynamic_tests)
target_sources(dcgm_stub_dynamic_tests PRIVATE DcgmStubDynamicLibraryTests.cpp)
target_link_libraries(dcgm_stub_dynamic_tests
    PRIVATE
        Catch2::Catch2WithMain
        dcgm_interface
        dcgm_stub
        test-utils)
set_target_properties(dcgm_stub_dynamic_tests PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${DCGM_STUB_DYNAMIC_TEST_DIR}"
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "$ORIGIN")
add_dependencies(dcgm_stub_dynamic_tests dcgm_stub_fake_dcgm)

if (CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
    catch_discover_tests(dcgm_stub_dynamic_tests EXTRA_ARGS --colour-mode ansi)
endif()
