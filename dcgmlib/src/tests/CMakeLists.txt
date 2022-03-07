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
find_package(Threads REQUIRED)
find_package(fmt REQUIRED)

include(CTest)
include(Catch)

if (BUILD_TESTING)

    add_executable(dcgmlibtests)
    target_sources(dcgmlibtests
        PRIVATE
            DcgmlibTestsMain.cpp
            CacheTests.cpp
            MigManagerTests.cpp
            ApiTests.cpp
            GpuInstanceTests.cpp
    )

    target_link_libraries(dcgmlibtests PRIVATE
            dcgmtest_interface
            common_protobuf_interface
            common_interface
            dcgm_interface
    )
    
    target_link_libraries(dcgmlibtests
        PRIVATE
            -Wl,--whole-archive
                modules_objects
                dcgm_common
                dcgm_logging
                dcgm_mutex
                dcgm_static_private
                transport_objects
                sdk_nvml_essentials_objects
                sdk_nvml_loader
            -Wl,--no-whole-archive
            common_protobuf_objects
            dcgm
            Catch2::Catch2
            ${CMAKE_THREAD_LIBS_INIT}
            fmt::fmt
            rt
            dl
    )
    
    if (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "x86_64")
        catch_discover_tests(dcgmlibtests EXTRA_ARGS --use-colour yes)
    endif()
endif()

