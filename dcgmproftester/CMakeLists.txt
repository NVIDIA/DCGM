# Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
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

add_library(dcgmproftester_interface INTERFACE)
target_sources(dcgmproftester_interface INTERFACE
    Arguments.h
    DcgmProfTester.h
    DcgmProfTesterAnyPtr.hpp
    DistributedCudaContext.h
    DistributedCudaContextRuntime.hpp
    DistributedCudaContextRuntimeAnyPtr.hpp
    DistributedCudaContextRuntimeAnyUniquePtr.hpp
    DistributedCudaContextRuntimeFactory.h
    DistributedCudaContextRuntimeFactoryAnyUniquePtr.hpp
    Entity.h
    PhysicalGpu.h
    PhysicalGpuAnyPtr.hpp
    Reporter.h)

target_include_directories(dcgmproftester_interface INTERFACE
    "${CMAKE_CURRENT_SOURCE_DIR}")

target_link_libraries(dcgmproftester_interface INTERFACE
    common_interface)

macro(define_dcgmproftester cuda_version)
    set(dcgmproftester_exec dcgmproftester${cuda_version})
    set(dcgmproftester_lib dcgmproftester_lib${cuda_version})

    target_sources(${dcgmproftester_lib}
        PRIVATE
            Arguments.cpp
            Arguments.h
            DcgmProfTester.cpp
            DcgmProfTester.h
            DcgmProfTesterAnyPtr.hpp
            DistributedCudaContext.cpp
            DistributedCudaContext.h
            DistributedCudaContextRuntime.cpp
            DistributedCudaContextRuntime.hpp
            DistributedCudaContextRuntimeAnyPtr.hpp
            DistributedCudaContextRuntimeAnyUniquePtr.hpp
            DistributedCudaContextRuntimeFactory.cpp
            DistributedCudaContextRuntimeFactory.h
            DistributedCudaContextRuntimeFactoryAnyUniquePtr.hpp
            Entity.cpp
            Entity.h
            PhysicalGpu.cpp
            PhysicalGpu.h
            PhysicalGpuAnyPtr.hpp
            Reporter.cpp
            Reporter.h)

    target_link_libraries(${dcgmproftester_lib}
        PUBLIC
            dcgm_cuda_worker${cuda_version}
            dcgmproftester_interface
        PRIVATE
            Threads::Threads
            common_interface
            cuda_ops_interface${cuda_version}
            dcgm
            dcgm_common
            dcgm_interface
            dcgm_logging
            dcgm_mutex
            dl
            fmt::fmt
            rt
            sdk_nvml_essentials_objects
            transport_objects)

    target_sources(${dcgmproftester_exec}
        PRIVATE
            DcgmProfTesterMain.cpp)

    target_link_options(${dcgmproftester_exec}
        PRIVATE
            -Wl,--version-script,${CMAKE_CURRENT_SOURCE_DIR}/proftester.linux_def)
    target_link_libraries(${dcgmproftester_exec}
        PRIVATE
            Threads::Threads
            common_interface
            dcgm
            dcgm_common
            dcgm_cuda_worker${cuda_version}
            dcgm_interface
            dcgm_logging
            dcgm_mutex
            ${dcgmproftester_lib}
            cublas_ops${cuda_version}
            cuda_ops${cuda_version}
            dl
            fmt::fmt
            rt
            sdk_nvml_essentials_objects
            transport_objects)
endmacro()

define_dcgmproftester(11)
define_dcgmproftester(12)
define_dcgmproftester(13)

add_subdirectory(mock)
add_subdirectory(tests)

install(
    FILES DcgmProfTesterKernels.ptx
    DESTINATION "${CMAKE_INSTALL_LIBEXECDIR}/${PROJECT_NAME}"
    COMPONENT dcgmproftesterkernels)

install(
    FILES DcgmProfTesterKernels.ptx
    DESTINATION "${DCGM_TESTS_APP_DIR}/${CMAKE_INSTALL_LIBEXECDIR}/${PROJECT_NAME}"
    COMPONENT Tests)
