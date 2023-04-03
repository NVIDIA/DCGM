# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
find_package(Yaml REQUIRED)
find_package(Jsoncpp REQUIRED)

include(CTest)
include(Catch)

if (BUILD_TESTING)

    add_executable(softwaretests)
    target_sources(softwaretests
        PRIVATE
            ../Software.cpp
            SoftwareTests.cpp
            SoftwareTestsMain.cpp
    )
    
    target_include_directories(softwaretests PRIVATE ..)
    
    target_link_libraries(softwaretests
        PRIVATE
            nvvs_without_main_objects
            Catch2::Catch2
    )
    
    target_link_libraries(softwaretests PRIVATE ${YAML_STATIC_LIBS} ${JSONCPP_STATIC_LIBS} rt dl)
    target_link_libraries(softwaretests PRIVATE ${CMAKE_THREAD_LIBS_INIT})
    target_link_options(softwaretests PRIVATE -rdynamic)

    if (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "x86_64")
        catch_discover_tests(softwaretests EXTRA_ARGS --use-colour yes)
    endif()
endif()
