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

add_subdirectory(DcgmThread)
add_subdirectory(serialize)
add_subdirectory(CudaWorker)
add_subdirectory(CudaLib)

add_subdirectory(transport)
add_subdirectory(mock)
add_subdirectory(tests)

configure_file(DcgmBuildInfo.cpp.in DcgmBuildInfo.cpp @ONLY)

target_include_directories(buildinfo_objects PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")

target_sources(buildinfo_objects PRIVATE
    "${CMAKE_CURRENT_BINARY_DIR}/DcgmBuildInfo.cpp")

target_include_directories(common_interface INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(common_interface INTERFACE
    dcgm_interface
    nvmli_interface
    sdk_nvml_interface)

target_sources(dcgm_logging PRIVATE
    DcgmLogging.cpp
    DcgmLogging.h
    DcgmLoggingImpl.h)

target_link_libraries(dcgm_logging PUBLIC
    common_interface
    fmt::fmt
    nvmli_interface)

target_sources(dcgm_mutex PRIVATE
    DcgmMutex.cpp
    DcgmMutex.h)

target_link_libraries(dcgm_mutex PUBLIC common_interface)

add_subdirectory(ChildProcess)
add_subdirectory(RemoteConn)

target_sources(dcgm_common PRIVATE
    CpuHelpers.cpp
    DcgmError.h
    DcgmFvBuffer.cpp
    DcgmFvBuffer.h
    DcgmGPUHardwareLimits.h
    DcgmGroupEntityPairHelpers.h
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
    DcgmUtilities.cpp
    DcgmUtilities.h
    DcgmWatchTable.cpp
    DcgmWatchTable.h
    DcgmWatcher.cpp
    DcgmWatcher.h
    EntityListHelpers.cpp
    EntityListHelpers.h
    FastPimpl.hpp
    FileSystemOperator.cpp
    FingerprintStore.cpp
    LsHw.cpp
    LsHw.h
    MigIdParser.cpp
    MigIdParser.hpp
    Semaphore.hpp
    Task.hpp
    TaskRunner.hpp
    ThreadSafeQueue.hpp
    TimeLib.cpp
    DcgmLib.cpp
    DcgmLib.h
    DcgmChildProcessManager.hpp
    DcgmChildProcessManager.cpp)

target_link_libraries(dcgm_common PUBLIC
    ${JSONCPP_STATIC_LIBS}
    childprocess
    buildinfo_objects
    common_interface
    dl
    fmt::fmt
    sdk_nvml_essentials_objects
    transport_objects)

target_sources(common_watch_objects PRIVATE
    DcgmError.h
    DcgmWatcher.cpp
    DcgmWatcher.h
    DcgmWatchTable.cpp
    DcgmWatchTable.h)

target_link_libraries(common_watch_objects PUBLIC common_interface)
