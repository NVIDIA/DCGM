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

macro(define_dcgmproftester cuda_version)
    set(dcgmproftester_exec dcgmproftester${cuda_version})

    target_compile_definitions(${dcgmproftester_exec}
        PRIVATE CUDA_VERSION_USED=${cuda_version})

    target_link_libraries(${dcgmproftester_exec}
        PRIVATE
            ${CMAKE_THREAD_LIBS_INIT}
            ${CUDA${cuda_version}_LIBS}
            ${CUDA${cuda_version}_STATIC_LIBS}
            common_interface
            dcgm
            dcgm_common
            dcgm_cublas_proxy${cuda_version}
            dcgm_cuda_worker${cuda_version}
            dcgm_interface
            dcgm_logging
            dcgm_mutex
            dl
            fmt::fmt
            rt
            sdk_nvml_essentials_objects
            transport_objects)

    target_include_directories(${dcgmproftester_exec}
        PRIVATE
            "${CUDA${cuda_version}_INCLUDE_DIR}")

    target_link_options(${dcgmproftester_exec}
        PRIVATE
            -Wl,--version-script,${CMAKE_CURRENT_SOURCE_DIR}/proftester.linux_def)

    target_sources(${dcgmproftester_exec}
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
            ${COMMON_SRCS})
endmacro()

define_dcgmproftester(11)
define_dcgmproftester(12)

install(
    FILES DcgmProfTesterKernels.ptx
    DESTINATION "${CMAKE_INSTALL_LIBEXECDIR}/${PROJECT_NAME}"
    COMPONENT Core)

install(
    FILES DcgmProfTesterKernels.ptx
    DESTINATION "${CMAKE_INSTALL_DATADIR}/dcgm_tests/apps/${CMAKE_INSTALL_LIBEXECDIR}/${PROJECT_NAME}"
    COMPONENT Tests)
