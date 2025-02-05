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

install(
    FILES  scripts/__init__.py
    DESTINATION "${DCGM_TESTS_INSTALL_DIR}/sdk_samples"
    COMPONENT Tests)

install(
    FILES
        scripts/dcgm_example.py
        scripts/__init__.py
    DESTINATION "${DCGM_TESTS_INSTALL_DIR}/sdk_samples/scripts"
    COMPONENT Tests)

install(
    DIRECTORY
        c_src
        scripts
        dcgm_diag_configs
    DESTINATION "src/${PROJECT_NAME}/sdk_samples"
    COMPONENT Development)

macro(dcgm_macro_build_sample sample)
  add_executable(${sample})
  target_sources(${sample} PRIVATE c_src/${sample}/${sample}.cpp)
  target_link_libraries(${sample} PRIVATE DCGM::dcgm)
  install(
    TARGETS ${sample}
    RUNTIME
        DESTINATION "${CMAKE_INSTALL_DATADIR}/dcgm_tests/apps/${DCGM_TESTS_ARCH}"
        COMPONENT Tests)
endmacro()

dcgm_macro_build_sample(configuration_sample)
dcgm_macro_build_sample(field_value_sample)
dcgm_macro_build_sample(health_sample)
dcgm_macro_build_sample(modules_sample)
dcgm_macro_build_sample(policy_sample)
dcgm_macro_build_sample(process_stats_sample)
