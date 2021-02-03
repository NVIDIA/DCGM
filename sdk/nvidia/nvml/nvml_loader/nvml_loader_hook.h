#ifndef __NVML_LOADER_HOOK_H__
#define __NVML_LOADER_HOOK_H__

#include <dcgm_nvml.h>

////////////////////////////////////////////////////////////////////////////////////////////////////////
// Declare API Hooking Mechanisms
#define NVML_ENTRY_POINT(nvmlFuncname, tsapiFuncname, argtypes, fmt, ...) \
    typedef nvmlReturn_t(*nvmlFuncname##_loader_t) argtypes;              \
    void set_##nvmlFuncname##Hook(nvmlFuncname##_loader_t nvmlFuncHook);  \
    void reset_##nvmlFuncname##Hook(void);

#include "entry_points.h"

////////////////////////////////////////////////////////////////////////////////////////////////////////
// Special Manually declared API Hooking Mechanisms

NVML_ENTRY_POINT(nvmlInit, nvmlInit, (void), "()")

NVML_ENTRY_POINT(nvmlInit_v2, nvmlInit_v2, (void), "()")

NVML_ENTRY_POINT(nvmlInitWithFlags, nvmlInitWithFlags, (unsigned int flags), "(0x%x)", flags)

NVML_ENTRY_POINT(nvmlShutdown, nvmlShutdown, (void), "()")

#undef NVML_ENTRY_POINT

////////////////////////////////////////////////////////////////////////////////////////////////////////
// Hooking Management API's
void resetAllNvmlHooks(void);

#endif /* __NVML_LOADER_HOOK_H__ */
