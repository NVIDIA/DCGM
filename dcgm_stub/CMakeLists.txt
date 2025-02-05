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

configure_file(version_config.h.in version_config.h @ONLY)
target_include_directories(dcgm_stub PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
target_sources(dcgm_stub PRIVATE dcgm_stub_static.c GetNthMapsToken.c)
target_link_libraries(dcgm_stub PRIVATE dcgm_interface nvmli_interface)
target_link_libraries(dcgm_stub PRIVATE dl rt)
target_compile_definitions(dcgm_stub PRIVATE DCGM_API_EXPORT=1)

add_subdirectory(tests)
