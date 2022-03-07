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
# DCGM_HOSTENGINE_LOCATION env var should point to the final nv-hostengine location on the system

if (${DCGM_PACKAGING})
    set(DCGM_HOSTENGINE_LOCATION /usr/${CMAKE_INSTALL_BINDIR})

    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/initctl/nvidia-dcgm.conf.in ${CMAKE_CURRENT_BINARY_DIR}/initctl/nvidia-dcgm.conf @ONLY NEWLINE_STYLE LF)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/systemd/dcgm.service.in ${CMAKE_CURRENT_BINARY_DIR}/systemd/dcgm.service @ONLY NEWLINE_STYLE LF)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/systemd/nvidia-dcgm.service.in ${CMAKE_CURRENT_BINARY_DIR}/systemd/nvidia-dcgm.service @ONLY NEWLINE_STYLE LF)

    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/initctl/nvidia-dcgm.conf DESTINATION "/etc/init" COMPONENT DCGM)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/systemd/dcgm.service DESTINATION "/usr/lib/systemd/system" COMPONENT DCGM)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/systemd/nvidia-dcgm.service DESTINATION "/usr/lib/systemd/system" COMPONENT DCGM)
endif()
