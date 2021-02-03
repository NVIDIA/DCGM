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
add_subdirectory(transport)
add_subdirectory(protobuf)
add_subdirectory(tests)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/DcgmBuildInfo.cpp.in ${CMAKE_CURRENT_BINARY_DIR}/DcgmBuildInfo.cpp @ONLY)

add_library(buildinfo_objects OBJECT)
target_include_directories(buildinfo_objects PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_sources(buildinfo_objects PRIVATE
    ${CMAKE_CURRENT_BINARY_DIR}/DcgmBuildInfo.cpp)

add_library(common_interface INTERFACE)
target_include_directories(common_interface INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(common_interface INTERFACE sdk_nvml_interface common_protobuf_interface dcgm_interface)

add_library(common_objects OBJECT)
target_sources(common_objects PRIVATE
    dcgm_errors.c
    dcgm_fields.cpp
    DcgmError.h
    DcgmFvBuffer.cpp
    DcgmFvBuffer.h
    DcgmGPUHardwareLimits.h
    DcgmLogging.cpp
    DcgmLogging.h
    DcgmMutex.cpp
    DcgmMutex.h
    DcgmPolicyRequest.cpp
    DcgmPolicyRequest.h
    DcgmRequest.cpp
    DcgmRequest.h
    DcgmSettings.cpp
    DcgmSettings.h
    DcgmStatCollection.cpp
    DcgmStatCollection.h
    DcgmStatus.cpp
    DcgmStatus.h
    DcgmStringConversions.h
    DcgmStringHelpers.cpp
    DcgmStringHelpers.h
    DcgmTaskRunner.cpp
    DcgmTaskRunner.h
    DcgmThread.cpp
    DcgmThread.h
    DcgmUtilities.cpp
    DcgmUtilities.h
    DcgmWatcher.cpp
    DcgmWatcher.h
    DcgmWatchTable.cpp
    DcgmWatchTable.h
    Semaphore.hpp
    Task.hpp
    TaskRunner.hpp
    ThreadSafeQueue.hpp
)

target_link_libraries(common_objects PUBLIC common_interface)
target_link_libraries(common_objects PUBLIC transport_objects buildinfo_objects)


add_library(common_watch_objects OBJECT)
target_sources(common_watch_objects PRIVATE
    dcgm_errors.c
    dcgm_fields.cpp
    DcgmError.h
    DcgmWatcher.cpp
    DcgmWatcher.h
    DcgmWatchTable.cpp
    DcgmWatchTable.h
)
target_link_libraries(common_watch_objects PUBLIC common_interface)
