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
add_executable(cuda_assert_64bit IMPORTED)
set_target_properties(cuda_assert_64bit PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/cuda_assert_64bit)

add_executable(cuda_ctx_create_64bit IMPORTED)
set_target_properties(
    cuda_ctx_create_64bit
    PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/cuda_ctx_create_64bit
)

install(
    PROGRAMS $<TARGET_PROPERTY:cuda_assert_64bit,IMPORTED_LOCATION>
    DESTINATION ${DCGM_TESTS_INSTALL_DIR}/apps/cuda_ctx_create
    COMPONENT Tests
)
install(
    PROGRAMS $<TARGET_PROPERTY:cuda_ctx_create_64bit,IMPORTED_LOCATION>
    DESTINATION ${DCGM_TESTS_INSTALL_DIR}/apps/cuda_ctx_create
    COMPONENT Tests
)
install(
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    DESTINATION ${DCGM_TESTS_INSTALL_DIR}/apps/cuda_ctx_create
    COMPONENT Tests
    FILES_MATCHING
    PATTERN "*.ptx"
)
