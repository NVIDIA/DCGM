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

install(CODE "
    file(INSTALL
        DESTINATION \"$<INSTALL_PREFIX>/${CMAKE_INSTALL_LIBEXECDIR}/${PROJECT_NAME}/cuda12/lib\"
        TYPE SHARED_LIBRARY
        FILES
            \"${CMAKE_CURRENT_SOURCE_DIR}/${CMAKE_SYSTEM_PROCESSOR}-cuda12/libcurand.so.10.3.10.19\")

    file(CREATE_LINK
        \"libcurand.so.10.3.10.19\"
        \"\$ENV{DESTDIR}$<INSTALL_PREFIX>/${CMAKE_INSTALL_LIBEXECDIR}/${PROJECT_NAME}/cuda12/lib/libcurand.so.10\"
        SYMBOLIC)

    file(CREATE_LINK
        \"libcurand.so.10\"
        \"\$ENV{DESTDIR}$<INSTALL_PREFIX>/${CMAKE_INSTALL_LIBEXECDIR}/${PROJECT_NAME}/cuda12/lib/libcurand.so\"
        SYMBOLIC)
    "
    COMPONENT CUDA12_CURAND)

install(CODE "
    file(INSTALL
        DESTINATION \"$<INSTALL_PREFIX>/${CMAKE_INSTALL_LIBEXECDIR}/${PROJECT_NAME}/cuda13/lib\"
        TYPE SHARED_LIBRARY
        FILES
            \"${CMAKE_CURRENT_SOURCE_DIR}/${CMAKE_SYSTEM_PROCESSOR}-cuda13/libcurand.so.10.4.0.35\")

    file(CREATE_LINK
        \"libcurand.so.10.4.0.35\"
        \"\$ENV{DESTDIR}$<INSTALL_PREFIX>/${CMAKE_INSTALL_LIBEXECDIR}/${PROJECT_NAME}/cuda13/lib/libcurand.so.10\"
        SYMBOLIC)

    file(CREATE_LINK
        \"libcurand.so.10\"
        \"\$ENV{DESTDIR}$<INSTALL_PREFIX>/${CMAKE_INSTALL_LIBEXECDIR}/${PROJECT_NAME}/cuda13/lib/libcurand.so\"
        SYMBOLIC)
    "
    COMPONENT CUDA13_CURAND)

install(CODE "
    file(INSTALL
        DESTINATION \"$<INSTALL_PREFIX>/${DCGM_TESTS_APP_DIR}/nvvs/plugins/cuda12\"
        TYPE SHARED_LIBRARY
        FILES
            \"${CMAKE_CURRENT_SOURCE_DIR}/${CMAKE_SYSTEM_PROCESSOR}-cuda12/libcurand.so.10.3.10.19\")

    file(CREATE_LINK
        \"libcurand.so.10.3.10.19\"
        \"\$ENV{DESTDIR}$<INSTALL_PREFIX>/${DCGM_TESTS_APP_DIR}/nvvs/plugins/cuda12/libcurand.so.10\"
        SYMBOLIC)

    file(CREATE_LINK
        \"libcurand.so.10\"
        \"\$ENV{DESTDIR}$<INSTALL_PREFIX>/${DCGM_TESTS_APP_DIR}/nvvs/plugins/cuda12/libcurand.so\"
        SYMBOLIC)

    file(MAKE_DIRECTORY
        \"\$ENV{DESTDIR}$<INSTALL_PREFIX>/${DCGM_TESTS_APP_DIR}/nvvs/plugins/cuda12/updated\")

    file(CREATE_LINK
        \"../libcurand.so.10.3.10.19\"
        \"\$ENV{DESTDIR}$<INSTALL_PREFIX>/${DCGM_TESTS_APP_DIR}/nvvs/plugins/cuda12/updated/libcurand.so.10\"
        SYMBOLIC)
    "
    COMPONENT Tests)

install(CODE "
    file(INSTALL
        DESTINATION \"$<INSTALL_PREFIX>/${DCGM_TESTS_APP_DIR}/nvvs/plugins/cuda13\"
        TYPE SHARED_LIBRARY
        FILES
            \"${CMAKE_CURRENT_SOURCE_DIR}/${CMAKE_SYSTEM_PROCESSOR}-cuda13/libcurand.so.10.4.0.35\")

    file(CREATE_LINK
        \"libcurand.so.10.4.0.35\"
        \"\$ENV{DESTDIR}$<INSTALL_PREFIX>/${DCGM_TESTS_APP_DIR}/nvvs/plugins/cuda13/libcurand.so.10\"
        SYMBOLIC)

    file(CREATE_LINK
        \"libcurand.so.10\"
        \"\$ENV{DESTDIR}$<INSTALL_PREFIX>/${DCGM_TESTS_APP_DIR}/nvvs/plugins/cuda13/libcurand.so\"
        SYMBOLIC)

    file(MAKE_DIRECTORY
        \"\$ENV{DESTDIR}$<INSTALL_PREFIX>/${DCGM_TESTS_APP_DIR}/nvvs/plugins/cuda13/updated\")

    file(CREATE_LINK
        \"../libcurand.so.10.4.0.35\"
        \"\$ENV{DESTDIR}$<INSTALL_PREFIX>/${DCGM_TESTS_APP_DIR}/nvvs/plugins/cuda13/updated/libcurand.so.10\"
        SYMBOLIC)
    "
    COMPONENT Tests)
