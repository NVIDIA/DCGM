/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
#ifndef __CUDA_LOADER_H__
#define __CUDA_LOADER_H__

#include <cuda.h>

typedef enum cudaLibraryLoadResult
{
    CUDA_LIBRARY_LOAD_SUCCESS = 0,
    CUDA_LIBRARY_ERROR_NOT_FOUND,
    CUDA_LIBRARY_ERROR_API_NOT_FOUND,
    CUDA_LIBRARY_ERROR_UNLOAD_FAILED,
    CUDA_LIBRARY_ERROR_OUT_OF_MEMORY
} cudaLibraryLoadResult_t;

#if defined(__cplusplus)
extern "C" {
#endif

cudaLibraryLoadResult_t loadDefaultCudaLibrary(void);
cudaLibraryLoadResult_t unloadCudaLibrary(void);

#ifdef __cplusplus
}
#endif

#endif
