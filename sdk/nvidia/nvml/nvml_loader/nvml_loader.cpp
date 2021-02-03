#include <dcgm_nvml.h>
#include "nvml_error_strings.h"
#include "nvml_loader_hook.h"
#include <dlfcn.h>
#include <mutex>

static void *g_nvmlLib                                      = 0;
static volatile unsigned int g_nvmlStaticLibResetHooksCount = 0;

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
        static volatile int isLookupDone = 0;                                                            \
        NVML_API_HOOK(libFunctionName, ##__VA_ARGS__);                                                   \
                                                                                                         \
        if (!g_nvmlLib)                                                                                  \
            return NVML_ERROR_UNINITIALIZED;                                                             \
                                                                                                         \
        if (!isLookupDone)                                                                               \
        {                                                                                                \
            static std::mutex initLock;                                                                  \
            std::lock_guard<std::mutex> guard(initLock);                                                 \
            if (!isLookupDone)                                                                           \
            {                                                                                            \
                libFunctionName##DefaultFunc                                                             \
                    = (libFunctionName##_loader_t)nvmlLoaderGetProcAddress(g_nvmlLib, #libFunctionName); \
                isLookupDone = 1;                                                                        \
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

static nvmlReturn_t nvmlLoadDefaultSharedLibrary(void)
{
    static std::mutex dcgmLibLock;

    if (g_nvmlLib)
        return NVML_ERROR_ALREADY_INITIALIZED;

    /* Lock our mutex */
    std::lock_guard<std::mutex> guard(dcgmLibLock);

#ifdef _UNIX
    g_nvmlLib = nvmlLoaderLoadLibrary("libnvidia-ml.so.1");
    if (g_nvmlLib)
        goto success;

    //
    // for Ubuntu we support /usr/lib{,32,64}/nvidia-current/...
    // However, we add these paths to ldconfig so this is not needed
    // If user messes with ldconfig after the installer sets things up it's up to them to fix apps
    //

    // For x64 .run and other installs
    g_nvmlLib = nvmlLoaderLoadLibrary("/usr/lib64/libnvidia-ml.so.1");
    if (g_nvmlLib)
        goto success;

    // For RPM Fusion (64 bit)
    g_nvmlLib = nvmlLoaderLoadLibrary("/usr/lib64/nvidia/libnvidia-ml.so.1");
    if (g_nvmlLib)
        goto success;

    // For some 32 bit and some 64 bit .run and other installs
    g_nvmlLib = nvmlLoaderLoadLibrary("/usr/lib/libnvidia-ml.so.1");
    if (g_nvmlLib)
        goto success;

    // For some 32 bit .run and other installs
    g_nvmlLib = nvmlLoaderLoadLibrary("/usr/lib32/libnvidia-ml.so.1");
    if (g_nvmlLib)
        goto success;

    // For RPM Fusion (32 bit)
    g_nvmlLib = nvmlLoaderLoadLibrary("/usr/lib/nvidia/libnvidia-ml.so.1");
    if (g_nvmlLib)
        goto success;

    return NVML_ERROR_LIBRARY_NOT_FOUND;

success:
    return NVML_SUCCESS;
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
