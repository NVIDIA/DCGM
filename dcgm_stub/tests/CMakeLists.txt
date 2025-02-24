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

add_executable(dcgm_stub_tests)
target_sources(dcgm_stub_tests PRIVATE
    test_maps_tokens.cpp
    ../GetNthMapsToken.c)

target_link_libraries(dcgm_stub_tests PRIVATE Catch2::Catch2WithMain)

if (CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
    catch_discover_tests(dcgm_stub_tests EXTRA_ARGS --colour-mode ansi)
endif()
