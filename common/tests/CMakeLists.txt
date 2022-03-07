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
set(CTEST_USE_LAUNCHERS 1)

find_package(Catch2 REQUIRED)

include(CTest)
include(Catch)

if (BUILD_TESTING)

    add_executable(commontests)
    target_sources(commontests
        PRIVATE
            CommonTestsMain.cpp
            SemaphoreTests.cpp
            TaskRunnerTests.cpp
            ThreadSafeQueueTests.cpp
            WatchTableTests.cpp
            BuildInfoTests.cpp
            StringHelpersTests.cpp
            DcgmUtilitiesTests.cpp
            TimeLibTests.cpp
    )

    find_package(Threads REQUIRED)
    
    target_link_libraries(commontests
        PRIVATE
            common_interface
            transport_objects
            dcgm
            dcgm_common
            dcgm_logging
            dcgm_mutex
            Catch2::Catch2
            sdk_nvml_essentials_objects
            sdk_nvml_interface
            sdk_nvml_loader
            ${CMAKE_THREAD_LIBS_INIT}
            rt
    )

    if (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "x86_64")
        catch_discover_tests(commontests EXTRA_ARGS --use-colour yes)
    endif()
endif()

