/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
#include <dcgm_nvml.h>
#include "nvml_error_strings.h"
#include "nvml_loader_hook.h"
#include <dlfcn.h>
#include <cstdlib>

#include <atomic>
#include <mutex>

static void *g_nvmlLib                                     = 0;
static std::atomic_uint32_t g_nvmlStaticLibResetHooksCount = 1;
#ifdef INJECTION_LIBRARY_AVAILABLE
static bool g_injectionLibraryLoaded = false;
#endif

// The following defines the hooking mechanism that calls the hooked function
// set by a user of this library. Insert this macro to enable an API to be hooked
// if the hooked function is set. This was done because nvmlInit() is a special
// case which must be hookable before it attempts to dynamically load a library.
#define NVML_API_HOOK(libFunctionName, ...)                                          \
    do                                                                               \
    {                                                                                \
        if ((NULL != libFunctionName##HookedFunc)                                    \
            && (libFunctionName##HookResetCount == g_nvmlStaticLibResetHooksCount))  \
        {                                                                            \
            nvmlReturn_t hookedFuncResult;                                           \
            /* The number of times this hook was reset equals the number of times */ \
            /* the static library was reset. So call the hooked function */          \
            hookedFuncResult = (*libFunctionName##HookedFunc)(__VA_ARGS__);          \
            return hookedFuncResult;                                                 \
        }                                                                            \
    } while (0)

#define NVML_DYNAMIC_WRAP(newName, libFunctionName, argtypes, ...)                                       \
    static libFunctionName##_loader_t libFunctionName##DefaultFunc = NULL;                               \
    static libFunctionName##_loader_t libFunctionName##HookedFunc  = NULL;                               \
    static volatile unsigned int libFunctionName##HookResetCount   = 0;                                  \
    void set_##libFunctionName##Hook(libFunctionName##_loader_t nvmlFuncHook)                            \
    {                                                                                                    \
        libFunctionName##HookResetCount = g_nvmlStaticLibResetHooksCount;                                \
        libFunctionName##HookedFunc     = nvmlFuncHook;                                                  \
    }                                                                                                    \
    void reset_##libFunctionName##Hook(void)                                                             \
    {                                                                                                    \
        libFunctionName##HookedFunc = NULL;                                                              \
    }                                                                                                    \
    nvmlReturn_t newName argtypes                                                                        \
    {                                                                                                    \
        static volatile uint32_t isLookupDone = 0;                                                       \
        NVML_API_HOOK(libFunctionName, ##__VA_ARGS__);                                                   \
                                                                                                         \
        if (!g_nvmlLib)                                                                                  \
            return NVML_ERROR_UNINITIALIZED;                                                             \
                                                                                                         \
        if (isLookupDone != g_nvmlStaticLibResetHooksCount)                                              \
        {                                                                                                \
            static std::mutex initLock;                                                                  \
            std::lock_guard<std::mutex> guard(initLock);                                                 \
            if (isLookupDone != g_nvmlStaticLibResetHooksCount)                                          \
            {                                                                                            \
                libFunctionName##DefaultFunc                                                             \
                    = (libFunctionName##_loader_t)nvmlLoaderGetProcAddress(g_nvmlLib, #libFunctionName); \
                isLookupDone = g_nvmlStaticLibResetHooksCount;                                           \
            }                                                                                            \
        }                                                                                                \
                                                                                                         \
        if (!libFunctionName##DefaultFunc)                                                               \
            return NVML_ERROR_FUNCTION_NOT_FOUND;                                                        \
                                                                                                         \
        return (*libFunctionName##DefaultFunc)(__VA_ARGS__);                                             \
    }

static void (*nvmlLoaderGetProcAddress(void *lib, const char *name))(void)
{
    return (void (*)(void))dlsym(lib, (const char *)name);
}

static void *nvmlLoaderLoadLibrary(const char *name)
{
    return dlopen(name, RTLD_NOW);
}

#define NVML_ENTRY_POINT(nvmlFuncname, tsapiFuncname, argtypes, fmt, ...) \
    NVML_DYNAMIC_WRAP(nvmlFuncname, nvmlFuncname, argtypes, ##__VA_ARGS__)
#include "entry_points.h"
#undef NVML_ENTRY_POINT

#ifdef INJECTION_LIBRARY_AVAILABLE
#define NVML_INJECTION_ENTRY_POINT(nvmlFuncname, tsapiFuncname, argtypes, fmt, ...) \
    NVML_DYNAMIC_WRAP(nvmlFuncname, nvmlFuncname, argtypes, ##__VA_ARGS__)
#include <entry_point_nvml_injection.h>
#undef NVML_INJECTION_ENTRY_POINT
#endif

static nvmlReturn_t nvmlLoadDefaultSharedLibrary(void)
{
    static std::mutex dcgmLibLock;

    if (g_nvmlLib)
        return NVML_ERROR_ALREADY_INITIALIZED;

#ifdef INJECTION_LIBRARY_AVAILABLE
    g_injectionLibraryLoaded = false;
#endif

    /* Lock our mutex */
    std::lock_guard<std::mutex> guard(dcgmLibLock);

#ifdef _UNIX
    const char *nvmlPaths[] =
    {
        //
        // for Ubuntu we support /usr/lib{,32,64}/nvidia-current/...
        // However, we add these paths to ldconfig so this is not needed
        // If user messes with ldconfig after the installer sets things up it's up to them to fix apps
        //
        "libnvidia-ml.so.1",
        // For x64 .run and other installs
        "/usr/lib64/libnvidia-ml.so.1",
        // For RPM Fusion (64 bit)
        "/usr/lib64/nvidia/libnvidia-ml.so.1",
        // For some 32 bit and some 64 bit .run and other installs
        "/usr/lib/libnvidia-ml.so.1",
        // For some 32 bit .run and other installs
        "/usr/lib32/libnvidia-ml.so.1",
        // For RPM Fusion (32 bit)
        "/usr/lib/nvidia/libnvidia-ml.so.1",
        nullptr
    };

    const char **paths = nvmlPaths;

#ifdef INJECTION_LIBRARY_AVAILABLE
    const char *nvmlInjectionPaths[] =
    {
        "libnvml_injection.so.1",
        "./apps/amd64/libnvml_injection.so.1",
        "./libnvml_injection.so.1",
        nullptr,
    };

    char *injectionMode = getenv("NVML_INJECTION_MODE");
    if (injectionMode != nullptr)
    {
        paths = nvmlInjectionPaths;
        g_injectionLibraryLoaded = true;
    }
#endif

    for (unsigned int i = 0; paths[i] != nullptr; i++)
    {
        g_nvmlLib = nvmlLoaderLoadLibrary(paths[i]);
        if (g_nvmlLib)
        {
            return NVML_SUCCESS;
        }
    }

    return NVML_ERROR_LIBRARY_NOT_FOUND;

#endif // _UNIX

#ifdef _WINDOWS
    return NVML_ERROR_NOT_SUPPORTED; /* Windows not supported by this loader */
#endif                               // _WINDOWS
}

static nvmlReturn_t localNvmlInit(void);
NVML_DYNAMIC_WRAP(localNvmlInit, nvmlInit, ())
nvmlReturn_t nvmlInit(void)
{
    NVML_API_HOOK(nvmlInit);

    if (!g_nvmlLib)
    {
        nvmlReturn_t result = nvmlLoadDefaultSharedLibrary();
        if (NVML_SUCCESS != result)
            return result;
    }

    return localNvmlInit();
}

static nvmlReturn_t localNvmlInit_v2(void);
NVML_DYNAMIC_WRAP(localNvmlInit_v2, nvmlInit_v2, ())
nvmlReturn_t nvmlInit_v2(void)
{
    NVML_API_HOOK(nvmlInit_v2);

    if (!g_nvmlLib)
    {
        nvmlReturn_t result = nvmlLoadDefaultSharedLibrary();
        if (NVML_SUCCESS != result)
            return result;
    }

    return localNvmlInit_v2();
}

static nvmlReturn_t localNvmlInitWithFlags(unsigned int flags);
NVML_DYNAMIC_WRAP(localNvmlInitWithFlags, nvmlInitWithFlags, (unsigned int flags), flags)
nvmlReturn_t nvmlInitWithFlags(unsigned int flags)
{
    NVML_API_HOOK(nvmlInitWithFlags, 0);

    if (!g_nvmlLib)
    {
        nvmlReturn_t result = nvmlLoadDefaultSharedLibrary();
        if (NVML_SUCCESS != result)
            return result;
    }

    return localNvmlInitWithFlags(flags);
}

NVML_DYNAMIC_WRAP(nvmlShutdown, nvmlShutdown, ())

typedef const char *(*nvmlErrorString_loader_t)(nvmlReturn_t result);
static const char *localNvmlErrorString(nvmlReturn_t result)
{
    static nvmlErrorString_loader_t func = NULL;
    static volatile int isLookupDone     = 0;

    if (!g_nvmlLib)
        return NULL;

    if (!isLookupDone)
    {
        static std::mutex initLock;
        std::lock_guard<std::mutex> guard(initLock);
        if (!isLookupDone)
        {
            func         = (nvmlErrorString_loader_t)nvmlLoaderGetProcAddress(g_nvmlLib, "nvmlErrorString");
            isLookupDone = 1;
        }
    }

    if (!func)
        return NULL;

    return func(result);
}

const char *nvmlErrorString(nvmlReturn_t result)
{
    const char *str = errorString(result);

    if (!str)
        str = localNvmlErrorString(result);

    if (!str)
        str = "Unknown Error";

    return str;
}

/**
 * Notifies all hooks that they are to reset the next time their associated NVML API is 
 * invoked using a global reset counter. API's compare their local counter to the global
 * count and reset their hooks if a mismatch is detected.
 */
void resetAllNvmlHooks(void)
{
    ++g_nvmlStaticLibResetHooksCount;
}

#ifdef INJECTION_LIBRARY_AVAILABLE
void nvmlClearLibraryHandleIfNeeded(void)
{
    bool needToClear          = false;
    const char *injectionMode = getenv("NVML_INJECTION_MODE");

    if (g_injectionLibraryLoaded && injectionMode == nullptr)
    {
        needToClear = true;
    }
    else if (g_injectionLibraryLoaded == false && injectionMode != nullptr)
    {
        needToClear = true;
    }
 
    if (needToClear && g_nvmlLib != 0)
    {
        dlclose(g_nvmlLib);
        g_nvmlLib = 0;
        resetAllNvmlHooks();
    }
}
#endif 

