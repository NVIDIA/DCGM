# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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
find_package(Protobuf REQUIRED)

protobuf_generate_cpp(COMMON_SRCS COMMON_HDRS
    dcgm.proto
    )

add_library(common_protobuf_gen STATIC)
target_sources(common_protobuf_gen PRIVATE ${COMMON_SRCS})
target_sources(common_protobuf_gen PRIVATE ${COMMON_HDRS})
target_include_directories(common_protobuf_gen PUBLIC ${Protobuf_INCLUDE_DIRS})
target_include_directories(common_protobuf_gen PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(common_protobuf_gen PUBLIC ${Protobuf_LIBRARIES})

add_library(common_protobuf_interface INTERFACE)
target_include_directories(common_protobuf_interface INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
    )
#add_dependencies(common_protobuf_interface common_protobuf_gen)
target_link_libraries(common_protobuf_interface INTERFACE common_protobuf_gen)

add_library(common_protobuf_objects OBJECT)
target_link_libraries(common_protobuf_objects PUBLIC common_protobuf_gen transport_objects common_protobuf_interface)
target_sources(common_protobuf_objects PRIVATE DcgmProtobuf.cpp)
target_sources(common_protobuf_objects PUBLIC DcgmProtobuf.h)
