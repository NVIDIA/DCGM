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
find_package(Cuda REQUIRED)
find_package(Threads REQUIRED)
find_package(fmt REQUIRED)

macro(define_dcgmproftester cuda_version)
    set(dcgmproftester_exec dcgmproftester${cuda_version})
    add_executable(${dcgmproftester_exec})
    list(APPEND DCGMPROFTESTER_TARGETS ${dcgmproftester_exec})
    target_compile_definitions(${dcgmproftester_exec} PRIVATE CUDA_VERSION_USED=${cuda_version})
    target_link_libraries(
        ${dcgmproftester_exec}
        PRIVATE
            dcgm
            dcgm_interface
            dcgm_logging
            dcgm_mutex
            common_interface
            transport_objects
            sdk_nvml_essentials_objects
            fmt::fmt
    )
    target_include_directories(${dcgmproftester_exec} PRIVATE ${CUDA${cuda_version}_INCLUDE_DIR})
    target_link_libraries(${dcgmproftester_exec} PRIVATE dl rt)
    target_link_libraries(${dcgmproftester_exec} PRIVATE dcgm_cublas_proxy${cuda_version})
    target_link_libraries(${dcgmproftester_exec} PRIVATE ${CUDA${cuda_version}_STATIC_LIBS})
    target_link_libraries(${dcgmproftester_exec} PRIVATE ${CUDA${cuda_version}_LIBS})
    target_sources(
        ${dcgmproftester_exec}
        PRIVATE
        DcgmProfTester.cpp
        DcgmProfTester.h
        DistributedCudaContext.cpp
        DistributedCudaContext.h
        Entity.h
        Entity.cpp
        PhysicalGpu.cpp
        PhysicalGpu.h
        Arguments.cpp
        Arguments.h
        Reporter.cpp
        Reporter.h
        ${COMMON_SRCS}
    )

    if (${cuda_version} GREATER 10)
        target_sources(${dcgmproftester_exec} PRIVATE DcgmDgemm.cpp)
    endif()

    find_package(Threads REQUIRED)
    target_link_libraries(${dcgmproftester_exec} PRIVATE ${CMAKE_THREAD_LIBS_INIT})
    target_link_options(${dcgmproftester_exec} PRIVATE -Wl,--version-script,${CMAKE_CURRENT_SOURCE_DIR}/proftester.linux_def)
endmacro()

if (NOT ${CMAKE_SYSTEM_PROCESSOR} STREQUAL "aarch64")
    define_dcgmproftester(10)
endif()
define_dcgmproftester(11)

set(DCGMPROFTESTER ${DCGMPROFTESTER_TARGETS} PARENT_SCOPE)

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/DcgmProfTesterKernels.ptx DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/DcgmProfTesterKernels.ptx DESTINATION bin COMPONENT DCGM)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/DcgmProfTesterKernels.ptx DESTINATION ${DCGM_TESTS_APP_DIR} COMPONENT Tests)
