/*
 * Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
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

#include "ContextCreatePlugin.h"
#include "DcgmHandle.h"
#include "DcgmRecorder.h"
#include "Plugin.h"
#include "PluginDevice.h"
#include "cuda.h"
#include "dcgm_structs.h"
#include "timelib.h"
#include <CudaCommon.h>
#include <CudaCommonAnyPtr.hpp>
#include <CudaOpsAnyUniquePtr.hpp>
#include <DcgmOpsAnyPtr.hpp>
#include <NvvsStructs.h>

#define CTX_CREATED 0x0
#define CTX_SKIP    0x1
#define CTX_FAIL    0x2

class ContextCreateDevice : public PluginDevice
{
public:
    CUdevice cuDevice {};
    CUcontext cuContext {};

    ContextCreateDevice(DcgmNs::DcgmOpsAnyPtr dcgmOps, CudaOpsAnyPtr cudaOps)
        : PluginDevice(dcgmOps, cudaOps)
    {}

    ContextCreateDevice(std::string const &testName,
                        unsigned int ndi,
                        const char *pciBusId,
                        Plugin *p,
                        DcgmHandle &handle,
                        DcgmNs::DcgmOpsAnyPtr dcgmOps,
                        CudaOpsAnyPtr cudaOps,
                        CudaCommonAnyPtr cudaCommon)
        : PluginDevice(testName, ndi, pciBusId, p, dcgmOps, handle.GetHandle(), cudaOps)
        , cuContext(0)
    {
        char buf[256] {};
        const char *errorString = nullptr;
        dcgmDeviceAttributes_t attr {};
        attr.version = dcgmDeviceAttributes_version3;

        dcgmReturn_t ret = dcgmOps->dcgmGetDeviceAttributes(handle.GetHandle(), ndi, &attr);

        if (ret != DCGM_ST_OK)
        {
            DcgmError d { gpuId };
            DCGM_ERROR_FORMAT_MESSAGE_DCGM(DCGM_FR_DCGM_API, d, ret, "dcgmGetDeviceAttributes");
            snprintf(buf, sizeof(buf), "for GPU %u", this->gpuId);
            d.AddDetail(buf);
            throw d;
        }

        CUresult cuSt = cudaOps->cuInit(0);

        if (cuSt)
        {
            DcgmError d { gpuId };
            DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_CUDA_API, d, "cuInit");
            cudaOps->cuGetErrorString(cuSt, &errorString);
            if (errorString != NULL)
            {
                snprintf(buf,
                         sizeof(buf),
                         ": '%s' (%d)%s",
                         errorString,
                         static_cast<int>(cuSt),
                         cudaCommon->GetAdditionalCuInitDetail(cuSt));
                d.AddDetail(buf);
            }
            throw d;
        }

        cuSt = cudaOps->cuDeviceGetByPCIBusId(&cuDevice, attr.identifiers.pciBusId);
        if (cuSt)
        {
            DcgmError d { gpuId };
            DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_CUDA_API, d, "cuDeviceGetByPCIBusId");
            cudaOps->cuGetErrorString(cuSt, &errorString);
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
    ContextCreate(TestParameters *testParameters,
                  ContextCreatePlugin *plugin,
                  dcgmHandle_t handle,
                  DcgmNs::DcgmOpsAnyPtr dcgmOps,
                  CudaOpsAnyPtr cudaOps,
                  CudaCommonAnyPtr cudaCommon);
    ~ContextCreate();

    /*************************************************************************/
    /*
     * Initialize devices and resources for this plugin.
     */
    std::string Init(const dcgmDiagPluginEntityList_v1 &entityList);

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
    int Run(const dcgmDiagPluginEntityList_v1 &entityList);

    /*************************************************************************/
    /*
     * Attempt to create a context for each GPU in the list
     *
     * @return:  0 on success
     *           1 for skipping
     *          -1 for failure to create
     */
    int CanCreateContext();

    /*************************************************************************/
    /*
     * Attempt to create a context for each GPU in the list
     * This is the templated version to be used for mocked cases
     *
     * @return:  0 on success
     *           1 for skipping
     *          -1 for failure to create
     */
    template <typename cuCtxCreateFunc, typename cuCtxDestroyFunc>
    int CanCreateContext(std::vector<std::unique_ptr<ContextCreateDevice>> &theDevices,
                         cuCtxCreateFunc cuCtxCreateImpl,
                         cuCtxDestroyFunc cuCtxDestroyImpl)
    {
        return CanCreateContext(
            theDevices, cuCtxCreateImpl, cuCtxDestroyImpl, [this](CUresult cuSt, const char **errStr) {
                return m_cudaOps->cuGetErrorString(cuSt, errStr);
            });
    }

    template <typename cuCtxCreateFunc, typename cuCtxDestroyFunc, typename cuGetErrorStringFunc>
    int CanCreateContext(std::vector<std::unique_ptr<ContextCreateDevice>> &theDevices,
                         cuCtxCreateFunc cuCtxCreateImpl,
                         cuCtxDestroyFunc cuCtxDestroyImpl,
                         cuGetErrorStringFunc cuGetErrorStringImpl)
    {
        int created = CTX_CREATED;
        CUresult cuSt;
        std::stringstream err;
        std::string error;

        for (size_t i = 0; i < theDevices.size(); i++)
        {
            cuSt = cuCtxCreateImpl(&theDevices[i]->cuContext, 0, theDevices[i]->cuDevice);

            if (cuSt == CUDA_SUCCESS)
            {
                cuCtxDestroyImpl(theDevices[i]->cuContext);
                continue;
            }
            else if (cuSt == CUDA_ERROR_UNKNOWN)
            {
                err << "GPU " << theDevices[i]->gpuId << " is in prohibted mode; skipping test.";
                m_plugin->AddInfo(m_plugin->GetCtxCreateTestName(), err.str());
                log_debug(err.str());
                created |= CTX_SKIP;
            }
            else
            {
                const char *errStr;
                cuGetErrorStringImpl(cuSt, &errStr);
                DcgmError d { theDevices[i]->gpuId };
                DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_CUDA_CONTEXT, d, theDevices[i]->gpuId, errStr);
                m_plugin->AddError(m_plugin->GetCtxCreateTestName(), d);
                log_error("{}", d.GetMessage());
                created |= CTX_FAIL;
            }
        }

        return created;
    }

private:
    /*************************************************************************/
    /*
     * GpusAreNonExclusive()
     *
     * Returns: true if the compute mode allows us to run this test
     */
    bool GpusAreNonExclusive();

    /*************************************************************************/
    ContextCreatePlugin *m_plugin;    /* Which plugin we're a part of. This is a paramter to the instance */
    TestParameters *m_testParameters; /* The test parameters for this run of NVVS */
    std::vector<std::unique_ptr<ContextCreateDevice>> m_device; /* Per-device data */
    DcgmRecorder *m_dcgmRecorder;
    DcgmHandle m_dcgmHandle;
    DcgmGroup m_dcgmGroup;
    DcgmNs::DcgmOpsAnyPtr m_dcgmOps; /* DCGM API interface */
    CudaOpsAnyPtr m_cudaOps;         //!< CUDA API interface.
    CudaCommonAnyPtr m_cudaCommon;
};

#endif
