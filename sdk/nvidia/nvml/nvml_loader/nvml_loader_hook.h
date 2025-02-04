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
#ifndef __NVML_LOADER_HOOK_H__
#define __NVML_LOADER_HOOK_H__

#include <dcgm_nvml.h>
#ifdef INJECTION_LIBRARY_AVAILABLE
#include <nvml_injection.h>
#endif

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

#ifdef INJECTION_LIBRARY_AVAILABLE
#define NVML_INJECTION_ENTRY_POINT(nvmlFuncname, tsapiFuncname, argtypes, fmt, ...) \
    typedef nvmlReturn_t(*nvmlFuncname##_loader_t) argtypes;              \
    void set_##nvmlFuncname##Hook(nvmlFuncname##_loader_t nvmlFuncHook);  \
    void reset_##nvmlFuncname##Hook(void);

#include <entry_point_nvml_injection.h>

#undef NVML_INJECTION_ENTRY_POINT
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////////
// Hooking Management API's
void resetAllNvmlHooks(void);

#endif /* __NVML_LOADER_HOOK_H__ */
