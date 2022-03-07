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
add_library(sdk_nvml_interface INTERFACE)
target_link_libraries(sdk_nvml_interface INTERFACE sdk_nvidia)

set(
    NVML_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/common
    ${CMAKE_CURRENT_SOURCE_DIR}/common/external/tinythread/source
    ${CMAKE_CURRENT_SOURCE_DIR}/nvsmi/stats
)

target_include_directories(sdk_nvml_interface INTERFACE ${NVML_INCLUDE_DIRS})

#target_include_directories(sdk_nvml_interface SYSTEM INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
#target_include_directories(sdk_nvml_interface SYSTEM INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/common)
#target_include_directories(
#    sdk_nvml_interface SYSTEM
#    INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/common/external/tinythread/source
#)
#target_include_directories(sdk_nvml_interface SYSTEM INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/nvsmi/stats)

# target_include_directories(sdk_nvml_interface INTERFACE /usr/local/cuda100/include)
# target_include_directories(sdk_nvml_interface INTERFACE /usr/local/cuda100/kernel/common/inc)

add_library(sdk_nvml_essentials_objects STATIC)
target_link_libraries(sdk_nvml_essentials_objects PUBLIC sdk_nvml_interface)
target_sources(
    sdk_nvml_essentials_objects
    PRIVATE
    common/hashtable.c
    common/keyedvector.c
    common/measurementcollection.c
    common/MurmurHash3.c
    common/nvcmvalue.c
    common/timelib.c
    common/timeseries.c
    common/logging.c
)

add_library(sdk_nvml_loader STATIC)
target_link_libraries(sdk_nvml_loader PUBLIC sdk_nvml_interface)
target_sources(
    sdk_nvml_loader
    PRIVATE
    nvml_loader/nvml_loader.cpp
    nvml_loader/nvml_loader_hook.h
)
target_compile_definitions(sdk_nvml_loader PUBLIC -DNVML_STATIC_IMPORT=1)
target_compile_definitions(sdk_nvml_loader PRIVATE -D__NVML_API_VERSION_INTERNAL)
