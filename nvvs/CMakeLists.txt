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

set(CMAKE_CXX_VISIBILITY_PRESET default)
set(CMAKE_C_VISIBILITY_PRESET default)
set(CMAKE_VISIBILITY_INLINES_HIDDEN 0)

add_library(nvvs_interface INTERFACE)
target_include_directories(nvvs_interface INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(nvvs_interface INTERFACE common_interface dcgm_interface)

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/nvvs.conf
        DESTINATION ${DCGM_NVVS_CONFIG_DIR}
        PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ
        COMPONENT DCGM
        )

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/diag-skus.yaml.in ${CMAKE_CURRENT_BINARY_DIR}/diag-skus.yaml @ONLY)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/diag-skus.yaml
        DESTINATION ${DCGM_NVVS_INSTALL_DIR}
        PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ
        COMPONENT Config
        )

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/diag-skus.yaml
        DESTINATION ${DCGM_TESTS_INSTALL_DIR}/apps/nvvs/
        PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ
        COMPONENT Tests
        )
