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
#ifndef __CUDA_HOOK_H__
#define __CUDA_HOOK_H__

#include "cuda-loader.h"
#include <cuda.h>

////////////////////////////////////////////////////////////////////////////////////////////////////////
// Hook API's
#define BEGIN_ENTRYPOINTS
#define END_ENTRYPOINTS
#define CUDA_API_ENTRYPOINT(cudaFuncname, entrypointApiFuncname, argtypes, fmt, ...) \
    typedef CUresult(CUDAAPI *cudaFuncname##_loader_t) argtypes;                     \
    void set_##cudaFuncname##Hook(cudaFuncname##_loader_t cudaFuncHook);             \
    void reset_##cudaFuncname##Hook(void);

#include "cuda-entrypoints.h"

#undef CUDA_API_ENTRYPOINT
#undef END_ENTRYPOINTS
#undef BEGIN_ENTRYPOINTS

////////////////////////////////////////////////////////////////////////////////////////////////////////
// Hook Managment

void resetAllCudaHooks(void);

#endif /* __CUDA_HOOK_H__ */
