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

install(
    PROGRAMS "${CMAKE_SYSTEM_PROCESSOR}-cuda12/ubergemm2"
    DESTINATION "${CMAKE_INSTALL_LIBEXECDIR}/${PROJECT_NAME}/cuda12/bin"
    COMPONENT CUDA12_UBERGEMM2)

install(
    PROGRAMS "${CMAKE_SYSTEM_PROCESSOR}-cuda13/ubergemm2"
    DESTINATION "${CMAKE_INSTALL_LIBEXECDIR}/${PROJECT_NAME}/cuda13/bin"
    COMPONENT CUDA13_UBERGEMM2)

install(
    PROGRAMS "${CMAKE_SYSTEM_PROCESSOR}-cuda12/ubergemm2"
    DESTINATION "${CMAKE_INSTALL_DATADIR}/dcgm_tests/apps/nvvs/cuda12/bin"
    COMPONENT Tests)

install(
    PROGRAMS "${CMAKE_SYSTEM_PROCESSOR}-cuda13/ubergemm2"
    DESTINATION "${CMAKE_INSTALL_DATADIR}/dcgm_tests/apps/nvvs/cuda13/bin"
    COMPONENT Tests)
