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
#include "cuda-loader.h"
#include "cuda-hook.h"
#include <assert.h>
#include <cuda.h>
#include <dlfcn.h>
#include <string.h>

////////////////////////////////////////////////////////////////////////////////////////////////////////
// Import Table
// ----------------------------------------
// The following declares and defines an import table for all dynamically loaded CUDA API's.

#define BEGIN_ENTRYPOINTS            \
    typedef struct itblCudaLibary_st \
    {
#define END_ENTRYPOINTS \
    }                   \
    itblCudaLibrary;
#define CUDA_API_ENTRYPOINT(cudaFuncname, entrypointApiFuncname, argtypes, fmt, ...) \
    CUresult(CUDAAPI *cudaFuncname) argtypes;

#include "cuda-entrypoints.h"

#undef CUDA_API_ENTRYPOINT
#undef END_ENTRYPOINTS
#undef BEGIN_ENTRYPOINTS

// The import table of the dynamically loaded CUDA library
static itblCudaLibrary g_itblDefaultCudaLibrary;

// The import table consisting of hooks that are to be called instead of the API's in the CUDA library
static itblCudaLibrary g_itblCudaLibrary;
static void *g_cudaLibrary = 0;

////////////////////////////////////////////////////////////////////////////////////////////////////////
// Function Stubs
// ----------------------------------------
// The following defines function stubs for each of the CUDA API's that can be dynamically loaded
// at runtime. These stubs allow programs to use a provided unmodified <cuda.h> within their programs
// and link against this static library to dynamically link with the CUDA library at runtime. Each
// function stub forwards the call to their CUDA API equivalent.

#define BEGIN_ENTRYPOINTS
#define END_ENTRYPOINTS
#define CUDA_API_ENTRYPOINT(cudaFuncname, entrypointApiFuncname, argtypes, fmt, ...) \
    CUresult CUDAAPI cudaFuncname argtypes                                           \
    {                                                                                \
        if (NULL != g_itblCudaLibrary.cudaFuncname)                                  \
            return (*(g_itblCudaLibrary.cudaFuncname))(__VA_ARGS__);                 \
        return (*(g_itblDefaultCudaLibrary.cudaFuncname))(__VA_ARGS__);              \
    }                                                                                \
    void set_##cudaFuncname##Hook(cudaFuncname##_loader_t cudaFuncHook)              \
    {                                                                                \
        g_itblCudaLibrary.cudaFuncname = cudaFuncHook;                               \
    }                                                                                \
    void reset_##cudaFuncname##Hook(void)                                            \
    {                                                                                \
        g_itblCudaLibrary.cudaFuncname = NULL;                                       \
    }

#include "cuda-entrypoints.h"

#undef CUDA_API_ENTRYPOINT
#undef END_ENTRYPOINTS
#undef BEGIN_ENTRYPOINTS

static void (*cudaLoaderGetProcAddress(void *lib, const char *name))(void)
{
    return (void (*)(void))dlsym(lib, (const char *)name);
}

static void *cudaLoaderLoadLibrary(const char *name)
{
    return dlopen(name, RTLD_NOW);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////
// Pre-emptive API Loading
// ----------------------------------------
// The following defines a function which Dynamically loads all CUDA API's into the import table.

#define BEGIN_ENTRYPOINTS                                    \
    static cudaLibraryLoadResult_t loadCudaLibraryApis(void) \
    {                                                        \
        assert(g_cudaLibrary);
#define CUDA_API_ENTRYPOINT(cudaFuncname, entrypointApiFuncname, argtypes, fmt, ...)                \
    assert(NULL == g_itblDefaultCudaLibrary.cudaFuncname);                                          \
    g_itblDefaultCudaLibrary.cudaFuncname                                                           \
        = (cudaFuncname##_loader_t)cudaLoaderGetProcAddress(g_cudaLibrary, #entrypointApiFuncname); \
    if (NULL == g_itblDefaultCudaLibrary.cudaFuncname)                                              \
    {                                                                                               \
        return CUDA_LIBRARY_ERROR_API_NOT_FOUND;                                                    \
    }
#define END_ENTRYPOINTS               \
    return CUDA_LIBRARY_LOAD_SUCCESS; \
    }

#include "cuda-entrypoints.h"

#undef END_ENTRYPOINTS
#undef CUDA_API_ENTRYPOINT
#undef BEGIN_ENTRYPOINTS

/**
 * Attempts to dynamically load the CUDA library from typical locations where the CUDA
 * library is found.
 *
 * @return
 *         - CUDA_LIBRARY_LOAD_SUCCESS      If the CUDA library was found and successfully loaded
 *         - CUDA_ERROR_LIBRARY_NOT_FOUND   If the CUDA library was not found or could not be loaded
 */
cudaLibraryLoadResult_t loadDefaultCudaLibrary(void)
{
    if (g_cudaLibrary)
        return CUDA_LIBRARY_LOAD_SUCCESS;

    assert(sizeof(g_itblDefaultCudaLibrary) == sizeof(g_itblCudaLibrary));
    assert(sizeof(g_itblCudaLibrary) == sizeof(itblCudaLibrary));
    memset(&g_itblDefaultCudaLibrary, 0, sizeof(g_itblDefaultCudaLibrary));
    memset(&g_itblCudaLibrary, 0, sizeof(g_itblCudaLibrary));
#ifdef _UNIX

    g_cudaLibrary = cudaLoaderLoadLibrary("libcuda.so.1");
    if (g_cudaLibrary)
        return loadCudaLibraryApis();

    //
    // for Ubuntu we support /usr/lib{,32,64}/nvidia-current/...
    // However, we add these paths to ldconfig so this is not needed
    // If user messes with ldconfig after the installer sets things up it's up to them to fix apps
    //

    // For x64 .run installs
    g_cudaLibrary = cudaLoaderLoadLibrary("/usr/lib64/libcuda.so.1");
    if (g_cudaLibrary)
        return loadCudaLibraryApis();

    // For RPM fusion x64
    g_cudaLibrary = cudaLoaderLoadLibrary("/usr/lib64/nvidia/libcuda.so.1");
    if (g_cudaLibrary)
        return loadCudaLibraryApis();

    // For some 32 and 64 bit installs
    g_cudaLibrary = cudaLoaderLoadLibrary("/usr/lib/libcuda.so.1");
    if (g_cudaLibrary)
        return loadCudaLibraryApis();

    // For some 32 bit installs
    g_cudaLibrary = cudaLoaderLoadLibrary("/usr/lib32/libcuda.so.1");
    if (g_cudaLibrary)
        return loadCudaLibraryApis();

    // For RPM Fusion 32 bit
    g_cudaLibrary = cudaLoaderLoadLibrary("/usr/lib/nvidia/libcuda.so.1");
    if (g_cudaLibrary)
        return loadCudaLibraryApis();

#endif /* _UNIX */
#ifdef _WINDOWS
    // Load from the system directory (a trusted location)
    g_cudaLibrary = cudaLoaderLoadLibrary("nvcuda.dll");
    if (g_cudaLibrary)
        return loadCudaLibraryApis();
#endif /* _WINDOWS */

    return CUDA_LIBRARY_ERROR_NOT_FOUND;
}

/**
 * Attempts to dynamically unload the CUDA library.
 *
 * @return
 *         - CUDA_LIBRARY_LOAD_SUCCESS          If the CUDA library was successfully unloaded
 *         - CUDA_ERROR_LIBRARY_UNLOAD_FAILED   If the CUDA library could not be unloaded
 */
cudaLibraryLoadResult_t unloadCudaLibrary(void)
{
    memset(&g_itblDefaultCudaLibrary, 0, sizeof(itblCudaLibrary));
    if (g_cudaLibrary)
    {
        if (dlclose(g_cudaLibrary))
            return CUDA_LIBRARY_ERROR_UNLOAD_FAILED;
        g_cudaLibrary = 0;
    }
    return CUDA_LIBRARY_LOAD_SUCCESS;
}

void resetAllCudaHooks(void)
{
    memset(&g_itblCudaLibrary, 0, sizeof(itblCudaLibrary));
}
