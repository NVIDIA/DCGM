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
#ifndef CONTEXTCREATE_H
#define CONTEXTCREATE_H

#include "DcgmHandle.h"
#include "DcgmRecorder.h"
#include "Plugin.h"
#include "PluginDevice.h"
#include "cuda.h"
#include "dcgm_structs.h"
#include "timelib.h"
#include <CudaCommon.h>
#include <NvvsStructs.h>

#define CTX_CREATED 0x0
#define CTX_SKIP    0x1
#define CTX_FAIL    0x2

class ContextCreateDevice : public PluginDevice
{
public:
    CUdevice cuDevice;
    CUcontext cuContext;

    ContextCreateDevice(unsigned int ndi, const char *pciBusId, Plugin *p, DcgmHandle &handle)
        : PluginDevice(ndi, pciBusId, p)
        , cuContext(0)
    {
        char buf[256] = { 0 };
        const char *errorString;
        dcgmDeviceAttributes_t attr;
        memset(&attr, 0, sizeof(attr));
        attr.version = dcgmDeviceAttributes_version2;

        dcgmReturn_t ret = dcgmGetDeviceAttributes(handle.GetHandle(), ndi, &attr);

        if (ret != DCGM_ST_OK)
        {
            DcgmError d { gpuId };
            DCGM_ERROR_FORMAT_MESSAGE_DCGM(DCGM_FR_DCGM_API, d, ret, "dcgmGetDeviceAttributes");
            snprintf(buf, sizeof(buf), "for GPU %u", this->gpuId);
            d.AddDetail(buf);
            throw d;
        }

        CUresult cuSt = cuInit(0);

        if (cuSt)
        {
            DcgmError d { gpuId };
            DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_CUDA_API, d, "cuInit");
            cuGetErrorString(cuSt, &errorString);
            if (errorString != NULL)
            {
                snprintf(buf,
                         sizeof(buf),
                         ": '%s' (%d)%s",
                         errorString,
                         static_cast<int>(cuSt),
                         GetAdditionalCuInitDetail(cuSt));
                d.AddDetail(buf);
            }
            throw d;
        }

        cuSt = cuDeviceGetByPCIBusId(&cuDevice, attr.identifiers.pciBusId);
        if (cuSt)
        {
            DcgmError d { gpuId };
            DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_CUDA_API, d, "cuDeviceGetByPCIBusId");
            cuGetErrorString(cuSt, &errorString);
            if (errorString != NULL)
            {
                snprintf(buf, sizeof(buf), ": '%s' (%d) for GPU %u", errorString, static_cast<int>(cuSt), this->gpuId);
                d.AddDetail(buf);
            }
            throw d;
        }
    }
};

class ContextCreate
{
public:
    ContextCreate(TestParameters *testParameters, Plugin *plugin, dcgmHandle_t handle);
    ~ContextCreate();

    /*************************************************************************/
    /*
     * Initialize devices and resources for this plugin.
     */
    std::string Init(const dcgmDiagPluginGpuList_t &gpuList);

    /*************************************************************************/
    /*
     * Clean up any resources allocated by this object, including memory and
     * file handles.
     */
    void Cleanup();

    /*************************************************************************/
    /*
     * Run ContextCreate tests
     *
     * Returns 0 on success
     *        <0 on failure or early exit
     */
    int Run(const dcgmDiagPluginGpuList_t &gpuList);

    /*************************************************************************/
    /*
     * Attempt to create a context for each GPU in the list
     *
     * @return:  0 on success
     *           1 for skipping
     *          -1 for failure to create
     */
    int CanCreateContext();

private:
    /*************************************************************************/
    /*
     * GpusAreNonExclusive()
     *
     * Returns: true if the compute mode allows us to run this test
     */
    bool GpusAreNonExclusive();

    /*************************************************************************/
    Plugin *m_plugin;                            /* Which plugin we're a part of. This is a paramter to the instance */
    TestParameters *m_testParameters;            /* The test parameters for this run of NVVS */
    std::vector<ContextCreateDevice *> m_device; /* Per-device data */
    DcgmRecorder *m_dcgmRecorder;
    DcgmHandle m_dcgmHandle;
    DcgmGroup m_dcgmGroup;
};

#endif
