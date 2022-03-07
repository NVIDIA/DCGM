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
#include "dcgm_agent.h"
#include "dcgm_fields.h"
#include "dcgm_structs.h"
#include "dcgm_structs_internal.h"
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>

static void *lib_handle = NULL;


static const char wrongLibraryMessage[]
    = "\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
      "WARNING:\n"
      "\n"
      "Wrong libdcgm.so installed\n"
      "\n"
      "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n";

static void dcgmCheckLinkedLibrary(void)
{
#if defined(NV_LINUX)
    FILE *f;
    char command_line[512];
    char buff[256];

    /**
     * Check if the commands (lsof, tr and cut) exists on the underlying platform
     */
    sprintf(command_line,
            "command -v lsof >/dev/null 2>&1 && command -v tr >/dev/null 2>&1"
            " && command -v cut >/dev/null 2>&1  && echo 'true' || echo 'false'");
    if (!(f = popen(command_line, "r")))
    {
        return;
    }

    while (fgets(buff, sizeof(buff), f) != NULL)
    {
        char *p;
        if ((p = strstr(buff, "false")) != NULL)
        {
            pclose(f);
            return;
        }
    }

    pclose(f);

    pid_t pid = getpid();
    sprintf(command_line, "lsof -p %d | tr -s ' ' | cut -d ' ' -f 9 ", pid);
    if (!(f = popen(command_line, "r")))
    {
        return;
    }

    while (fgets(buff, sizeof(buff), f) != NULL)
    {
        char *p;
        if ((p = strstr(buff, "libdcgm")) != NULL)
        {
            printf("Linked to libdcgm library at wrong path : %s\n", buff);
            break;
        }
    }

    pclose(f);
#endif
}

#define DCGM_DYNAMIC_WRAP(newName, libFunctionName, argtypes, ...) \
    dcgmReturn_t newName argtypes                                  \
    {                                                              \
        dcgmReturn_t(*fn) argtypes;                                \
        if (lib_handle)                                            \
        {                                                          \
            fn = dlsym(lib_handle, __FUNCTION__);                  \
            return (*fn)(__VA_ARGS__);                             \
        }                                                          \
        else                                                       \
        {                                                          \
            printf("%s", wrongLibraryMessage);                     \
            return DCGM_ST_UNINITIALIZED;                          \
        }                                                          \
    }

#define DCGM_ENTRY_POINT(dcgmFuncname, tsapiFuncname, argtypes, fmt, ...) \
    DCGM_DYNAMIC_WRAP(dcgmFuncname, dcgmFuncname, argtypes, ##__VA_ARGS__)

//#define DCGM_INT_ENTRY_POINT(dcgmFuncname, tsapiFuncname, argtypes, fmt, ...)
#include "entry_point.h"
//#undef DCGM_INT_ENTRY_POINT
#undef DCGM_ENTRY_POINT

dcgmReturn_t dcgmInit(void)
{
    dcgmReturn_t (*fn)(void);
    lib_handle = dlopen("libdcgm.so.2", RTLD_LAZY);
    if (lib_handle)
    {
        fn = dlsym(lib_handle, "dcgmInit");
        return (*fn)();
    }
    else
    {
        printf("%s", wrongLibraryMessage);
        dcgmCheckLinkedLibrary();
        return DCGM_ST_UNINITIALIZED;
    }
}

dcgmReturn_t DCGM_PUBLIC_API dcgmStartEmbedded(dcgmOperationMode_t opMode, dcgmHandle_t *pDcgmHandle)
{
    dcgmReturn_t (*fn)(dcgmOperationMode_t, dcgmHandle_t *);
    lib_handle = dlopen("libdcgm.so.2", RTLD_LAZY);
    if (lib_handle)
    {
        fn = dlsym(lib_handle, "dcgmStartEmbedded");
        return (*fn)(opMode, pDcgmHandle);
    }
    else
    {
        printf("%s", wrongLibraryMessage);
        dcgmCheckLinkedLibrary();
        return DCGM_ST_UNINITIALIZED;
    }
}

dcgmReturn_t DCGM_PUBLIC_API dcgmStopEmbedded(dcgmHandle_t pDcgmHandle)
{
    dcgmReturn_t (*fn)(dcgmHandle_t);
    lib_handle = dlopen("libdcgm.so.2", RTLD_LAZY);
    if (lib_handle)
    {
        fn = dlsym(lib_handle, "dcgmStopEmbedded");
        return (*fn)(pDcgmHandle);
    }
    else
    {
        printf("%s", wrongLibraryMessage);
        dcgmCheckLinkedLibrary();
        return DCGM_ST_UNINITIALIZED;
    }
}

dcgmReturn_t DCGM_PUBLIC_API dcgmShutdown(void)
{
    dcgmReturn_t dcgmResult;
    dcgmReturn_t (*fn)(void);
    if (lib_handle)
    {
        fn         = dlsym(lib_handle, "dcgmShutdown");
        dcgmResult = (*fn)();
    }
    else
    {
        printf("%s", wrongLibraryMessage);
        return DCGM_ST_UNINITIALIZED;
    }
    if (lib_handle)
        dlclose(lib_handle);
    return dcgmResult;
}

#if 0  
// left here in case we ever want to include internal apis... must also include dcgm_*interal.h above
DCGM_DYNAMIC_WRAP(dcgmInternalGetExportTable, dcgmInternalGetExportTable,
        (const void **ppExportTable, const dcgmUuid_t *pExportTableId),
        ppExportTable, pExportTableId)
#endif

const char *dcgmErrorString(dcgmReturn_t result)
{
    return wrongLibraryMessage;
}
