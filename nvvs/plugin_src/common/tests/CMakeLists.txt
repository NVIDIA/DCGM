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

add_executable(plugincommontests)
target_sources(plugincommontests
    PRIVATE
        PluginCommonTests.cpp
)

# NOTE: linking dcgmlib is not allowed for this library. Instead, make a mock implementation
#       for any libdcgm function that is needed.
target_link_libraries(plugincommontests
    PRIVATE
        nvvs_without_main_objects
        Catch2::Catch2WithMain
        ${YAML_STATIC_LIBS}
        rt
)
target_link_libraries(plugincommontests
    PUBLIC
        nvvs_plugins_interface
        nvvs_plugins_common_objects
        nvvs_interface
)

if (CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
    catch_discover_tests(plugincommontests EXTRA_ARGS --colour-mode ansi)
endif()
