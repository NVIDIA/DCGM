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
set(CTEST_USE_LAUNCHERS 1)

find_package(Catch2 REQUIRED)
find_package(Jsoncpp REQUIRED)

include(CTest)
include(Catch)

if (BUILD_TESTING)

    add_executable(dcgmitests)
    target_sources(dcgmitests
        PRIVATE
        DcgmiTestsMain.cpp
        DiagTests.cpp
        MigTests.cpp
        OutputTests.cpp
        MigTestsHelper.cpp
    )

    # NOTE: linking dcgmlib is not allowed for this library. Instead, make a mock implementation
    #       for any libdcgm function that is needed.    
    target_link_libraries(dcgmitests
        PRIVATE
            dcgmi_interface
            dcgmi_objects
            nvvs_interface
            diag_interface
            common_objects
            buildinfo_objects
            sdk_nvml_essentials_objects
            sdk_nvml_interface
            Catch2::Catch2
            dcgm
            dcgmmodulediag
    )
    target_include_directories(dcgmitests PRIVATE ${JSONCPP_INCLUDE_DIR})
    target_link_libraries(dcgmitests PRIVATE ${CMAKE_THREAD_LIBS_INIT})
    target_link_options(dcgmitests PRIVATE -rdynamic)
    target_link_libraries(dcgmitests PRIVATE ${JSONCPP_STATIC_LIBS})

    if (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "x86_64")
        catch_discover_tests(dcgmitests EXTRA_ARGS --use-colour yes)
    endif()
endif()
