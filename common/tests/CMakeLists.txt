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

add_executable(commontests)
target_sources(commontests
    PRIVATE
        SemaphoreTests.cpp
        TaskRunnerTests.cpp
        ThreadSafeQueueTests.cpp
        WatchTableTests.cpp
        BuildInfoTests.cpp
        StringHelpersTests.cpp
        DcgmUtilitiesTests.cpp
        TimeLibTests.cpp
        DcgmLogging.cpp
        EntityListHelpersTests.cpp
        MigTestsHelper.cpp
        WildcardTests.cpp
        MigTests.cpp
        CpuHelpersTests.cpp
        LsHwTests.cpp
        FingerprintStoreTests.cpp
        HangDetectTests.cpp
        HangDetectHandlerTests.cpp
        HangDetectMonitorTests.cpp
        MockFileSystemOperatorTests.cpp
        mocks/MockChildProcess.cpp
        DcgmChildProcessManagerTests.cpp
        UriParserTests.cpp
        PermissionCheckTests.cpp
        DcgmMutexTests.cpp
)

target_link_libraries(commontests
    PRIVATE
        Catch2::Catch2WithMain
        common_interface
        transport_objects
        childprocess
        dcgm
        dcgm_common
        dcgm_logging
        dcgm_mutex
        sdk_nvml_essentials_objects
        sdk_nvml_interface
        sdk_nvml_loader
        mock_interface
        ${CMAKE_THREAD_LIBS_INIT}
        fmt::fmt
        rt
)

add_executable(hangdetectworkflow_tests)
target_sources(hangdetectworkflow_tests
    PRIVATE
        HangDetectWorkflowTests.cpp
)

target_link_libraries(hangdetectworkflow_tests
    PRIVATE
        Catch2::Catch2WithMain
        common_interface
        dcgm
        dcgm_common
        dcgm_mutex
        dcgm_logging
        fmt::fmt
        ${CMAKE_THREAD_LIBS_INIT}
        rt
        task_context_manager
)

if (CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
    catch_discover_tests(commontests
        PROPERTIES SKIP_REGULAR_EXPRESSION "[1-9][0-9]* skipped"
        EXTRA_ARGS --colour-mode ansi)
    catch_discover_tests(hangdetectworkflow_tests
        PROPERTIES SKIP_REGULAR_EXPRESSION "[1-9][0-9]* skipped"
        EXTRA_ARGS --colour-mode ansi)
endif()
