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

target_include_directories(transport_interface INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(transport_objects PUBLIC
    transport_interface
    dcgm_interface
    common_interface
    dcgm_thread_interface
    sdk_nvml_interface
    )

target_include_directories(transport_objects SYSTEM PUBLIC ${LIBEVENT_INCLUDE_DIR})
target_link_libraries(transport_objects PUBLIC ${LIBEVENT_STATIC_LIBS})
target_link_libraries(transport_objects PRIVATE dcgm_thread)

target_sources(transport_objects PRIVATE
    DcgmProtocol.cpp
    DcgmIpc.cpp
    )

target_sources(transport_objects PUBLIC
    DcgmProtocol.h
    DcgmIpc.h
    )

target_include_directories(transport_objects PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
