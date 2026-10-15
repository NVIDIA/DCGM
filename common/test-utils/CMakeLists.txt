# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

add_library(test-utils STATIC)

target_sources(test-utils
    PRIVATE
        MigHierarchyUtils.cpp
        ChildProcessUtils.hpp
        ScopedSignalHandler.hpp
        ScopedSignalMask.hpp
        SignalUtils.hpp
)

target_include_directories(test-utils
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}"
)

target_link_libraries(test-utils
    PUBLIC
        Catch2::Catch2
        common_interface
        fmt::fmt
)

add_subdirectory(tests)
