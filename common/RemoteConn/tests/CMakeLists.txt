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

add_executable(remoteconntests)

target_sources(remoteconntests PRIVATE
    MockChildProcess.hpp
    MockChildProcess.cpp
    MockChildProcessTests.cpp
    TestHelpers.hpp
    TestHelpers.cpp
    TcpSSHTunnelManagerMockTests.cpp
    UdsSSHTunnelManagerMockTests.cpp
)

target_link_libraries(remoteconntests
    PRIVATE
    remoteconn
    dcgm_common
    Catch2::Catch2WithMain
)

add_executable(remoteconnsystemtests)

target_sources(remoteconnsystemtests PRIVATE
    TestHelpers.hpp
    TestHelpers.cpp
    TcpSSHTunnelManagerTests.cpp
    UdsSSHTunnelManagerTests.cpp
)

target_link_libraries(remoteconnsystemtests
    PRIVATE
    remoteconn
    dcgm_common
    Catch2::Catch2WithMain
)

if (CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
    catch_discover_tests(remoteconntests EXTRA_ARGS --colour-mode ansi)
endif ()
