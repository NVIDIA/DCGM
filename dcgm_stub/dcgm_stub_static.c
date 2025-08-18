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
#include "GetNthMapsToken.h"

#include <version_config.h>

#include <dcgm_agent.h>
#include <dcgm_fields.h>
#include <dcgm_multinode_internal.h>
#include <dcgm_structs.h>
#include <dcgm_structs_internal.h>
#include <nvml_injection_structs.h>

#include <dlfcn.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>


static void *lib_handle = NULL;

static const char *s_libName              = DCGM_LIB_SONAME;
static const char s_wrongLibraryMessage[] = "Wrong version of the DCGM library is linked:";
static const size_t s_libPathColumn       = 6;

static void printSharedLibraryPath(FILE *outStream, char const *libName)
{
    char fileName[512];
    sprintf(fileName, "/proc/%d/maps", getpid());

    FILE *fMaps = fopen(fileName, "r");
    if (NULL == fMaps)
    {
        fprintf(outStream, "Failed to open %s\n", fileName);
        return;
    }

    // Read the file line by line
    bool found = false;
    char line[1024];
    while (fgets(line, sizeof(line), fMaps))
    {
        char *libPath = GetNthMapsToken(line, s_libPathColumn);
        if (libPath == NULL)
        {
            continue;
        }
        char *p = strstr(libPath, libName);
        if (p == NULL)
        {
            continue;
        }
        fprintf(outStream, "Linked to the %s at path %s\n", libName, libPath);
        found = true;
        break;
    }

    if (!found)
    {
        fprintf(outStream, "Unable to find %s\n", libName);
    }
    fclose(fMaps);
}


#define DCGM_DYNAMIC_WRAP(newName, libFunctionName, argtypes, ...)                     \
    dcgmReturn_t newName argtypes                                                      \
    {                                                                                  \
        dcgmReturn_t(*fn) argtypes;                                                    \
        if (lib_handle)                                                                \
        {                                                                              \
            fn = dlsym(lib_handle, __FUNCTION__);                                      \
            if (fn == NULL)                                                            \
            {                                                                          \
                fprintf(stderr, "Failed to find %s in %s\n", __FUNCTION__, s_libName); \
                printSharedLibraryPath(stderr, s_libName);                             \
                return DCGM_ST_UNINITIALIZED;                                          \
            }                                                                          \
            return (*fn)(__VA_ARGS__);                                                 \
        }                                                                              \
        else                                                                           \
        {                                                                              \
            fprintf(stderr, "%s\n", s_wrongLibraryMessage);                            \
            printSharedLibraryPath(stderr, s_libName);                                 \
            return DCGM_ST_UNINITIALIZED;                                              \
        }                                                                              \
    }

#define DCGM_ENTRY_POINT(dcgmFuncname, tsapiFuncname, argtypes, fmt, ...) \
    DCGM_DYNAMIC_WRAP(dcgmFuncname, dcgmFuncname, argtypes, ##__VA_ARGS__)

#include "entry_point.h"

#undef DCGM_ENTRY_POINT


dcgmReturn_t dcgmInit(void)
{
    dcgmReturn_t (*fn)(void);
    lib_handle = dlopen(s_libName, RTLD_LAZY);
    if (lib_handle)
    {
        fn = dlsym(lib_handle, "dcgmInit");
        if (fn == NULL)
        {
            fprintf(stderr, "Failed to find dcgmInit in %s\n", s_libName);
            printSharedLibraryPath(stderr, s_libName);
            return DCGM_ST_UNINITIALIZED;
        }
        return (*fn)();
    }

    fprintf(stderr, "%s\n", s_wrongLibraryMessage);
    printSharedLibraryPath(stderr, s_libName);
    return DCGM_ST_UNINITIALIZED;
}

dcgmReturn_t DCGM_PUBLIC_API dcgmStartEmbedded(dcgmOperationMode_t opMode, dcgmHandle_t *pDcgmHandle)
{
    dcgmReturn_t (*fn)(dcgmOperationMode_t, dcgmHandle_t *);
    lib_handle = dlopen("libdcgm.so.4", RTLD_LAZY);
    if (lib_handle)
    {
        fn = dlsym(lib_handle, "dcgmStartEmbedded");
        if (fn == NULL)
        {
            fprintf(stderr, "Failed to find dcgmStartEmbedded in %s\n", s_libName);
            printSharedLibraryPath(stderr, s_libName);
            return DCGM_ST_UNINITIALIZED;
        }
        return (*fn)(opMode, pDcgmHandle);
    }

    fprintf(stderr, "%s\n", s_wrongLibraryMessage);
    printSharedLibraryPath(stderr, s_libName);
    return DCGM_ST_UNINITIALIZED;
}

dcgmReturn_t DCGM_PUBLIC_API dcgmStopEmbedded(dcgmHandle_t pDcgmHandle)
{
    dcgmReturn_t (*fn)(dcgmHandle_t);
    lib_handle = dlopen("libdcgm.so.4", RTLD_LAZY);
    if (lib_handle)
    {
        fn = dlsym(lib_handle, "dcgmStopEmbedded");
        if (fn == NULL)
        {
            fprintf(stderr, "Failed to find dcgmStopEmbedded in %s\n", s_libName);
            printSharedLibraryPath(stderr, s_libName);
            return DCGM_ST_UNINITIALIZED;
        }
        return (*fn)(pDcgmHandle);
    }

    fprintf(stderr, "%s\n", s_wrongLibraryMessage);
    printSharedLibraryPath(stderr, s_libName);
    return DCGM_ST_UNINITIALIZED;
}

dcgmReturn_t DCGM_PUBLIC_API dcgmShutdown(void)
{
    dcgmReturn_t dcgmResult;
    dcgmReturn_t (*fn)(void);
    if (lib_handle)
    {
        fn = dlsym(lib_handle, "dcgmShutdown");
        if (fn == NULL)
        {
            fprintf(stderr, "Failed to find dcgmShutdown in %s\n", s_libName);
            printSharedLibraryPath(stderr, s_libName);
            return DCGM_ST_UNINITIALIZED;
        }
        dcgmResult = (*fn)();
        dlclose(lib_handle);
        return dcgmResult;
    }

    fprintf(stderr, "%s\n", s_wrongLibraryMessage);
    printSharedLibraryPath(stderr, s_libName);
    return DCGM_ST_UNINITIALIZED;
}

#if 0  
// left here in case we ever want to include internal apis... must also include dcgm_*interal.h above
DCGM_DYNAMIC_WRAP(dcgmInternalGetExportTable, dcgmInternalGetExportTable,
        (const void **ppExportTable, const dcgmUuid_t *pExportTableId),
        ppExportTable, pExportTableId)
#endif

const char *errorString(dcgmReturn_t result)
{
    const char *(*fn)(dcgmReturn_t);
    if (lib_handle)
    {
        fn = dlsym(lib_handle, "errorString");
        if (fn == NULL)
        {
            fprintf(stderr, "Failed to find errorString in %s\n", s_libName);
            printSharedLibraryPath(stderr, s_libName);
            return NULL;
        }
        return (*fn)(result);
    }
    fprintf(stderr, "%s\n", s_wrongLibraryMessage);
    printSharedLibraryPath(stderr, s_libName);
    return NULL;
}

unsigned int dcgmCpuHierarchyCpuOwnsCore(unsigned int coreId, dcgmCpuHierarchyOwnedCores_v1 const *ownedCores)
{
    unsigned int (*fn)(unsigned int, dcgmCpuHierarchyOwnedCores_v1 const *);
    if (lib_handle)
    {
        fn = dlsym(lib_handle, "dcgmCpuHierarchyCpuOwnsCore");
        if (fn == NULL)
        {
            fprintf(stderr, "Failed to find dcgmCpuHierarchyCpuOwnsCore in %s\n", s_libName);
            printSharedLibraryPath(stderr, s_libName);
            return 0;
        }
        return (*fn)(coreId, ownedCores);
    }
    fprintf(stderr, "%s\n", s_wrongLibraryMessage);
    printSharedLibraryPath(stderr, s_libName);
    return 0;
}
