/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <DcgmLogging.h>
#include <DcgmOpsAnyPtr.hpp>
#include <dcgm_structs.h>

/********************************************************************/
/**
 * Gets a CUDA major version compatible with the given GPU architecture.
 *
 * @param gpuId GPU ID used for logging
 * @param chipArchitecture GPU architecture reported by DCGM
 * @param cudaDriverMajorVersion Current CUDA driver major version
 * @param cudaDriverMinorVersion Current CUDA driver minor version
 * @return CUDA major version to use for CUDA-linked binaries
 */
inline unsigned int GetCompatibleCudaMajorVersionForGpu(unsigned int gpuId,
                                                        dcgmChipArchitecture_t chipArchitecture,
                                                        unsigned int cudaDriverMajorVersion,
                                                        unsigned int cudaDriverMinorVersion)
{
    // CUDA 13.0 drops support for everything that is < 7.5 SM Cuda Compatibility. These older SKUs need to run
    // a 12.9-linked application.
    // Note: We only check one GPU because NVVS can only run on homogenous GPUs.
    if ((chipArchitecture == DCGM_CHIP_ARCH_MAXWELL || chipArchitecture == DCGM_CHIP_ARCH_PASCAL
         || chipArchitecture == DCGM_CHIP_ARCH_VOLTA)
        && cudaDriverMajorVersion == 13 && cudaDriverMinorVersion == 0)
    {
        log_info("Using CUDA 12 for GPU {} because it is Maxwell, Pascal, or Volta and CUDA 13.0 is detected", gpuId);
        return 12;
    }

    return cudaDriverMajorVersion;
}

/********************************************************************/
/**
 * Gets a CUDA major version compatible with the given GPU after reading its chip architecture.
 *
 * @param dcgmOps DCGM operations used to read GPU architecture
 * @param handle DCGM handle
 * @param gpuId GPU ID to check
 * @param cudaDriverMajorVersion Current CUDA driver major version
 * @param cudaDriverMinorVersion Current CUDA driver minor version
 * @return CUDA major version to use for CUDA-linked binaries
 */
inline unsigned int GetCompatibleCudaMajorVersionForGpu(DcgmNs::DcgmOpsAnyPtr dcgmOps,
                                                        dcgmHandle_t handle,
                                                        unsigned int gpuId,
                                                        unsigned int cudaDriverMajorVersion,
                                                        unsigned int cudaDriverMinorVersion)
{
    if (cudaDriverMajorVersion == 0)
    {
        log_warning("User provided CUDA version is 0. Using default.");
        return cudaDriverMajorVersion;
    }

    dcgmChipArchitecture_t chipArchitecture {};
    dcgmReturn_t const ret = dcgmOps->dcgmGetGpuChipArchitecture(handle, gpuId, &chipArchitecture);
    if (ret != DCGM_ST_OK)
    {
        log_warning(
            "Unable to get GPU chip architecture for GPU {}, error: {}. Using provided cudaDriverMajorVersion: {}",
            gpuId,
            ret,
            cudaDriverMajorVersion);
        return cudaDriverMajorVersion;
    }

    return GetCompatibleCudaMajorVersionForGpu(gpuId, chipArchitecture, cudaDriverMajorVersion, cudaDriverMinorVersion);
}
