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

if (NOT BUILD_TESTING)
    return()
endif()

add_executable(mndiag_tests)

target_sources(mndiag_tests
    PRIVATE
        MnDiagCommonTests.cpp
        MpiRunnerTests.cpp
        MnDiagMpiRunnerTests.cpp
        MnDiagStateMachineTests.cpp
        MnDiagProcessUtilsTests.cpp
        MnDiagManagerTests.cpp
)

target_include_directories(mndiag_tests 
    PRIVATE 
        ${CMAKE_CURRENT_SOURCE_DIR}/..
)

# NOTE: linking dcgmlib is not allowed for this library. Instead, make a mock implementation
#       for any libdcgm function that is needed.
# Link essential libraries
target_link_libraries(mndiag_tests
    PRIVATE
        dcgm
        dcgmmodulemndiag_private_static
        dcgmmodulemndiag_common
        dcgm_common
        dcgm_logging
        Catch2::Catch2WithMain
        fmt::fmt
        dcgm_interface
        dcgm  # This should contain the errorString implementation
        modules_interface
        dcgmmodulemndiag_private_static  # Contains MnDiagStateMachine implementation
        sdk_nvml_essentials_objects  # For timelib_usecSince1970
        config_interface
        dcgm_mutex
        remoteconn
        modules_objects
        mndiag_interface
)

# Add only necessary system libraries
target_link_libraries(mndiag_tests PRIVATE ${CMAKE_THREAD_LIBS_INIT})

if (CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
    catch_discover_tests(mndiag_tests EXTRA_ARGS --colour-mode ansi)
endif()